"""
Training loop for MapEncoder.

Task   : given (obstacle_map, source_mask), predict the BFS distance field
         from the source to every reachable free cell, normalised to [0, 1].
Loss   : MSE over free cells only (obstacle cells masked out).
Metric : mean free-cell MSE on the validation set, logged to wandb.
"""
import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
import wandb.errors

from .config import MapEncoderConfig
from .encoder import MapEncoder
from .dataset import MapEncoderDataset

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DistanceFieldNet(nn.Module):
    """
    MapEncoder + small prediction head for the distance field task.

    The encoder produces a (D, H, W) feature map at full spatial resolution.
    The head maps those D channels down to 1 channel (the predicted distance field).
    No upsampling needed — the encoder never reduces spatial resolution.

    After training, only encoder weights are carried forward to downstream tasks.
    """

    def __init__(self, encoder: MapEncoder, out_channels: int):
        super().__init__()
        self.encoder = encoder
        D = out_channels
        self.head = nn.Sequential(
            nn.Conv2d(D, D // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(D // 2, 1, kernel_size=1),
            nn.Sigmoid(),   # output in [0, 1] matches the normalised distance target
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) -> (B, 1, H, W) predicted distance field."""
        return self.head(self.encoder(x))


def build_model(cfg: MapEncoderConfig) -> DistanceFieldNet:
    e = cfg.encoder
    encoder = MapEncoder(
        in_channels=e.in_channels,
        base_channels=e.base_channels,
        out_channels=e.out_channels,
        n_dilated_layers=e.n_dilated_layers,
    )
    return DistanceFieldNet(encoder, e.out_channels)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE computed only over cells where mask == 1 (free cells)."""
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: DistanceFieldNet,
    epoch: int,
    val_loss: float,
    cfg: MapEncoderConfig,
):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "config": cfg,
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[DistanceFieldNet, int, MapEncoderConfig]:
    ckpt = torch.load(path, map_location=device)
    cfg: MapEncoderConfig = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["epoch"], cfg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: MapEncoderConfig):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    try:
        run = wandb.init(
            entity="unitbv-rovis",
            project="MapEncoder",
            settings=wandb.Settings(init_timeout=240),
            config={
                "map_width":        cfg.train.map_width,
                "map_height":       cfg.train.map_height,
                "num_train":        cfg.train.num_train,
                "num_val":          cfg.train.num_val,
                "batch_size":       cfg.train.batch_size,
                "lr":               cfg.train.lr,
                "epochs":           cfg.train.epochs,
                "base_channels":    cfg.encoder.base_channels,
                "out_channels":     cfg.encoder.out_channels,
                "n_dilated_layers": cfg.encoder.n_dilated_layers,
            },
        )
    except wandb.errors.CommError as e:
        print(f"W&B server sync timed out.")
        exit(1)

    print("Generating training data...")
    train_ds = MapEncoderDataset(
        cfg.train.num_train, cfg.train.map_width, cfg.train.map_height, seed=None,
    )
    val_ds = MapEncoderDataset(
        cfg.train.num_val, cfg.train.map_width, cfg.train.map_height, seed=42,
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr_min,
    )

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0

        for inp, target, mask in train_loader:
            inp, target, mask = inp.to(device), target.to(device), mask.to(device)
            loss = _masked_mse(model(inp), target, mask)
            optimizer.zero_grad()
            loss.backward()
            if cfg.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        log = {"epoch": epoch, "train_loss": avg_train, "lr": optimizer.param_groups[0]["lr"]}

        if epoch % cfg.train.val_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inp, target, mask in val_loader:
                    inp, target, mask = inp.to(device), target.to(device), mask.to(device)
                    val_loss += _masked_mse(model(inp), target, mask).item()

            avg_val = val_loss / len(val_loader)
            log["val_loss"] = avg_val
            print(f"Epoch {epoch:4d} | train {avg_train:.6f} | val {avg_val:.6f}")

            if epoch % cfg.train.checkpoint_every == 0:
                _save_checkpoint(model, epoch, avg_val, cfg)
        else:
            print(f"Epoch {epoch:4d} | train {avg_train:.6f}")

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

    test_ds = MapEncoderDataset(500, cfg.train.map_width, cfg.train.map_height, seed=99)
    loader  = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    with torch.no_grad():
        for inp, target, mask in loader:
            total_loss += _masked_mse(model(inp), target, mask).item()

    print(f"Test MSE (free cells only): {total_loss / len(loader):.6f}")
