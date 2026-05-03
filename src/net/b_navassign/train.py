"""
Training loop for BNavAssign (Bottleneck NavAssign).

Task   : given N robot positions and N target positions on an obstacle map,
         predict the 1-to-1 assignment that minimises the MAXIMUM individual
         travel cost (makespan) — the time until all targets are simultaneously
         occupied, since all robots move in parallel.
Loss   : weighted BCE between predicted doubly-stochastic matrix and ground-truth
         permutation matrix. pos_weight = (N-1) balances the 3:1 zero/one ratio.
Metric : val BCE loss, per-robot assignment accuracy (argmax match), cost ratio
         (predicted max cost / bottleneck-optimal max cost).

NOTE: Ground truth is currently a STUB — linear_sum_assignment (Hungarian) is
used as a placeholder. It must be replaced with the bottleneck assignment solver
(brute-force over N! permutations for small N, or a dedicated solver for large N)
before training produces meaningful results for the makespan objective.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import wandb.errors
from torch.utils.data import DataLoader

from .config import BNavAssignConfig
from .dataset import NavAssignDataset
from .model import NavAssignNet, build_model

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: NavAssignNet,
    epoch: int,
    val_loss: float,
    cfg: BNavAssignConfig,
):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch:04d}.pt")
    torch.save({
        "epoch":       epoch,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "config":      cfg,
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[NavAssignNet, int, BNavAssignConfig]:
    ckpt = torch.load(path, map_location=device)
    cfg: BNavAssignConfig = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["epoch"], cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _assignment_accuracy(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Fraction of robots whose predicted target matches the ground-truth target.

    Parameters
    ----------
    pred : (B, N, N) doubly-stochastic matrix
    gt   : (B, N, N) permutation matrix

    Returns
    -------
    float in [0, 1]
    """
    pred_hard = pred.argmax(dim=-1)   # (B, N)
    gt_hard   = gt.argmax(dim=-1)     # (B, N)
    return (pred_hard == gt_hard).float().mean().item()


def _cost_ratio(
    pred:        torch.Tensor,   # (B, N, N) doubly-stochastic matrix
    cost_matrix: torch.Tensor,   # (B, N, N) BFS distances
    gt:          torch.Tensor,   # (B, N, N) permutation matrix
) -> float:
    """
    Mean ratio of predicted makespan to bottleneck-optimal makespan.

    Makespan = max individual travel cost across all N robots.
    A value of 1.0 means the model always finds the optimal (min-max) assignment.
    A value of 1.05 means predicted assignments have a makespan 5% worse on average.

    Parameters
    ----------
    pred        : (B, N, N) doubly-stochastic matrix
    cost_matrix : (B, N, N) BFS pairwise distances
    gt          : (B, N, N) ground-truth permutation matrix (bottleneck optimal)
    """
    pred_hard = pred.argmax(dim=-1)   # (B, N) -- predicted target per robot
    gt_hard   = gt.argmax(dim=-1)     # (B, N) -- optimal target per robot

    B, N = pred_hard.shape
    robot_idx = torch.arange(N, device=pred_hard.device).unsqueeze(0).expand(B, N)

    b_idx = torch.arange(B, device=pred_hard.device).unsqueeze(1).expand(B, N)

    pred_cost = cost_matrix[b_idx, robot_idx, pred_hard].max(dim=-1).values  # (B,)
    opt_cost  = cost_matrix[b_idx, robot_idx, gt_hard  ].max(dim=-1).values  # (B,)

    # Only include scenarios where the optimal makespan is non-trivial (> 0).
    # Dividing by near-zero opt_cost produces meaningless ratios.
    valid = opt_cost > 0
    if valid.sum() == 0:
        return 1.0
    return (pred_cost[valid] / opt_cost[valid]).mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: BNavAssignConfig):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    try:
        run = wandb.init(
            entity="unitbv-rovis",
            project="Bottleneck-NavAssign",
            settings=wandb.Settings(init_timeout=240),
            config={
                "num_robots":            cfg.train.num_robots,
                "map_width":             cfg.map.width,
                "map_height":            cfg.map.height,
                "num_train":             cfg.train.num_train,
                "num_val":               cfg.train.num_val,
                "batch_size":            cfg.train.batch_size,
                "lr":                    cfg.train.lr,
                "epochs":                cfg.train.epochs,
                "encoder_freeze_epochs": cfg.train.encoder_freeze_epochs,
                "gnn_hidden_dim":        cfg.model.gnn_hidden_dim,
                "gnn_layers":            cfg.model.gnn_layers,
                "sinkhorn_iters":        cfg.model.sinkhorn_iters,
                "encoder_checkpoint":    cfg.model.encoder_checkpoint,
            },
        )
    except wandb.errors.CommError:
        raise

    print("Generating training data...")
    train_ds = NavAssignDataset(
        cfg.train.num_train, cfg.map.width, cfg.map.height,
        cfg.train.num_robots, seed=cfg.train.train_seed,
    )
    val_ds = NavAssignDataset(
        cfg.train.num_val, cfg.map.width, cfg.map.height,
        cfg.train.num_robots, seed=42,
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)

    # Freeze encoder for the first encoder_freeze_epochs epochs.
    # Only GNN + score MLP params are optimised initially.
    model.encoder.requires_grad_(False)
    non_encoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.Adam(non_encoder_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr_min,
    )

    encoder_unfrozen = False

    for epoch in range(1, cfg.train.epochs + 1):

        # Unfreeze encoder after encoder_freeze_epochs epochs.
        # Add encoder params as a separate param group with a reduced LR (0.1x)
        # to avoid destroying the pre-trained weights.
        if not encoder_unfrozen and epoch > cfg.train.encoder_freeze_epochs:
            model.encoder.requires_grad_(True)
            optimizer.add_param_group({
                "params":       list(model.encoder.parameters()),
                "lr":           cfg.train.lr * 0.1,
                "weight_decay": cfg.train.weight_decay,
            })
            encoder_unfrozen = True
            print(f"  Epoch {epoch}: encoder unfrozen (lr={cfg.train.lr * 0.1:.2e})")

        model.train()
        if not encoder_unfrozen:
            model.encoder.eval()   # keep BN running stats frozen while encoder is frozen
        total_loss = 0.0
        total_acc  = 0.0

        for obs, masks, r_pos, t_pos, gt, _ in train_loader:
            obs, masks  = obs.to(device),   masks.to(device)
            r_pos, t_pos = r_pos.to(device), t_pos.to(device)
            gt          = gt.to(device)

            pred, _ = model(obs, masks, r_pos, t_pos)  # (B, N, N), (B, N, N)
            # BCE on Sinkhorn output: gradient respects both row and column constraints.
            # weight tensor balances 3:1 zero/one ratio: ones get weight (N-1), zeros get 1.
            N = pred.shape[1]
            w = gt * (N - 1) + (1 - gt)                     # (B, N, N): 3 where gt=1, 1 where gt=0
            loss = F.binary_cross_entropy(pred, gt, weight=w)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_acc  += _assignment_accuracy(pred.detach(), gt)

        avg_train     = total_loss / len(train_loader)
        avg_train_acc = total_acc  / len(train_loader)
        log = {"epoch": epoch, "train_loss": avg_train, "train_acc": avg_train_acc, "lr": optimizer.param_groups[0]["lr"]}

        if epoch % cfg.train.val_every == 0:
            model.eval()
            val_loss  = 0.0
            val_acc   = 0.0
            val_ratio = 0.0

            with torch.no_grad():
                for obs, masks, r_pos, t_pos, gt, cost_mat in val_loader:
                    obs, masks   = obs.to(device),      masks.to(device)
                    r_pos, t_pos = r_pos.to(device),    t_pos.to(device)
                    gt           = gt.to(device)
                    cost_mat     = cost_mat.to(device)

                    pred, _ = model(obs, masks, r_pos, t_pos)
                    N        = pred.shape[1]
                    w        = gt * (N - 1) + (1 - gt)
                    val_loss  += F.binary_cross_entropy(pred, gt, weight=w).item()
                    val_acc   += _assignment_accuracy(pred, gt)
                    val_ratio += _cost_ratio(pred, cost_mat, gt)

            avg_val   = val_loss  / len(val_loader)
            avg_acc   = val_acc   / len(val_loader)
            avg_ratio = val_ratio / len(val_loader)
            log["val_loss"]       = avg_val
            log["val_acc"]        = avg_acc
            log["val_cost_ratio"] = avg_ratio
            print(f"Epoch {epoch:4d} | train {avg_train:.5f} (acc {avg_train_acc:.3f}) | "
                  f"val {avg_val:.5f} (acc {avg_acc:.3f}, cost_ratio {avg_ratio:.4f})")

            if epoch % cfg.train.checkpoint_every == 0:
                _save_checkpoint(model, epoch, avg_val, cfg)
        else:
            print(f"Epoch {epoch:4d} | train {avg_train:.5f} (acc {avg_train_acc:.3f})")

        scheduler.step()
        run.log(log, step=epoch)

    run.finish()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str):
    device = torch.device("cpu")
    model, epoch, cfg = load_checkpoint(checkpoint_path, device)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch}")

    test_ds = NavAssignDataset(
        1_000, cfg.map.width, cfg.map.height,
        cfg.train.num_robots, seed=99,
    )
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_acc  = 0.0

    total_ratio = 0.0

    with torch.no_grad():
        for obs, masks, r_pos, t_pos, gt, cost_mat in loader:
            pred, _ = model(obs, masks, r_pos, t_pos)
            N        = pred.shape[1]
            w        = gt * (N - 1) + (1 - gt)
            total_loss  += F.binary_cross_entropy(pred, gt, weight=w).item()
            total_acc   += _assignment_accuracy(pred, gt)
            total_ratio += _cost_ratio(pred, cost_mat, gt)

    print(f"Test BCE loss  : {total_loss  / len(loader):.5f}")
    print(f"Test accuracy  : {total_acc   / len(loader):.3f}")
    print(f"Test cost ratio: {total_ratio / len(loader):.4f}  (1.0 = always optimal)")
