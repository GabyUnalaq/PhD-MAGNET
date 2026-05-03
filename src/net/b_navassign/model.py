"""
NavAssignNet: N-robot to N-target optimal assignment network.

Architecture
------------
1. N encoder forward passes (one per robot):
      encoder(obstacle_map, robot_i_mask) -> F_i  (B, D, H, W)

2. Feature extraction via MapEncoder.extract():
      robot i node  : F_i at robot i's position          -> (B, D)
      target j node : mean_i [ F_i at target j's pos ]   -> (B, D)
      edge (i->j)   : F_i at target j's position         -> (B, D)

   The edge feature F_i[:, r_j, c_j] is a learned proxy for the A* cost
   from robot i to target j -- it encodes robot-i-centric spatial context
   at target j's location.

3. Project all features from D (encoder dim) to H (GNN hidden dim).

4. Bipartite GNN (gnn_layers rounds):
      Robot -> Target messages:  for each target j, aggregate over robots i
      Target -> Robot messages:  for each robot i, aggregate over targets j
   Edge features are fixed (not updated) across GNN rounds.

5. Score matrix: S[i,j] = MLP(h_robot_i || h_target_j)  -> (B, N, N)

6. Sinkhorn normalisation -> doubly-stochastic assignment matrix (B, N, N)
   Enforces both row and column sums = 1 (valid assignment).
"""
import torch
import torch.nn as nn

from src.net.map_encoder.encoder import MapEncoder
from .config import ModelConfig, BNavAssignConfig


# ---------------------------------------------------------------------------
# Sinkhorn normalisation
# ---------------------------------------------------------------------------

def sinkhorn(scores: torch.Tensor, n_iters: int) -> torch.Tensor:
    """
    Convert a raw score matrix to a doubly-stochastic matrix.

    Uses log-space iteration for numerical stability.

    Parameters
    ----------
    scores  : (B, N, N) raw scores (logits)
    n_iters : number of Sinkhorn iterations

    Returns
    -------
    (B, N, N) doubly-stochastic matrix (rows and columns sum to 1)
    """
    log_s = scores
    for _ in range(n_iters):
        log_s = log_s - log_s.logsumexp(dim=-1, keepdim=True)   # normalise rows
        log_s = log_s - log_s.logsumexp(dim=-2, keepdim=True)   # normalise cols
    return log_s.exp()


# ---------------------------------------------------------------------------
# Bipartite GNN layer
# ---------------------------------------------------------------------------

class _BipartiteGNNLayer(nn.Module):
    """
    One round of bidirectional message passing between robot and target nodes.

    Edge features are fixed inputs (not updated).

    Parameters
    ----------
    hidden_dim : H, feature dimension for robot/target nodes and edges
    dropout    : applied after each MLP
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        H = hidden_dim

        # Robot -> Target: message from robot i to target j
        # input: concat(robot_h[i], edge_h[i,j])  shape: 2H
        self.mlp_r2t = nn.Sequential(
            nn.Linear(2 * H, H), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        # Update target node: concat(target_h[j], aggregated_msg)  shape: 2H -> H
        self.proj_t = nn.Linear(2 * H, H)
        self.ln_t   = nn.LayerNorm(H)

        # Target -> Robot: message from target j to robot i
        # input: concat(target_h[j], edge_h[i,j])  shape: 2H
        self.mlp_t2r = nn.Sequential(
            nn.Linear(2 * H, H), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        # Update robot node: concat(robot_h[i], aggregated_msg)  shape: 2H -> H
        self.proj_r = nn.Linear(2 * H, H)
        self.ln_r   = nn.LayerNorm(H)

    def forward(
        self,
        robot_h:  torch.Tensor,   # (B, N, H)
        target_h: torch.Tensor,   # (B, N, H)
        edge_h:   torch.Tensor,   # (B, N, N, H)  -- [b, i, j] = robot i -> target j
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, H = robot_h.shape

        # --- Robot -> Target ---
        # Expand robot features to (B, N, N, H): same robot i for all targets j
        r_exp     = robot_h.unsqueeze(2).expand(B, N, N, H)           # (B, N, N, H)
        r2t_msgs  = self.mlp_r2t(torch.cat([r_exp, edge_h], dim=-1))  # (B, N, N, H)
        agg_r2t   = r2t_msgs.mean(dim=1)                               # (B, N, H) -- mean over robots
        target_h  = self.ln_t(target_h + self.proj_t(torch.cat([target_h, agg_r2t], dim=-1)))

        # --- Target -> Robot ---
        # Expand target features to (B, N, N, H): same target j for all robots i
        t_exp     = target_h.unsqueeze(1).expand(B, N, N, H)           # (B, N, N, H)
        t2r_msgs  = self.mlp_t2r(torch.cat([t_exp, edge_h], dim=-1))   # (B, N, N, H)
        agg_t2r   = t2r_msgs.mean(dim=2)                                # (B, N, H) -- mean over targets
        robot_h   = self.ln_r(robot_h + self.proj_r(torch.cat([robot_h, agg_t2r], dim=-1)))

        return robot_h, target_h


# ---------------------------------------------------------------------------
# NavAssignNet
# ---------------------------------------------------------------------------

class NavAssignNet(nn.Module):
    """
    Parameters
    ----------
    encoder : pre-trained MapEncoder (weights loaded from checkpoint)
    enc_dim : encoder output channels D (read from checkpoint config)
    cfg     : ModelConfig
    """

    def __init__(self, encoder: MapEncoder, enc_dim: int, cfg: ModelConfig):
        super().__init__()
        H = cfg.gnn_hidden_dim

        self.encoder       = encoder
        self.sinkhorn_iters = cfg.sinkhorn_iters

        # Project all encoder features (robot, target, edge) from D -> H
        self.feat_proj = nn.Linear(enc_dim, H)

        # Bipartite GNN
        self.gnn = nn.ModuleList([
            _BipartiteGNNLayer(H, cfg.dropout)
            for _ in range(cfg.gnn_layers)
        ])

        # Score MLP: concat(h_robot_i, h_target_j) -> scalar
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, 1),
        )

    def forward(
        self,
        obstacle_map:     torch.Tensor,   # (B, H, W)
        robot_masks:      torch.Tensor,   # (B, N, H, W)
        robot_positions:  torch.Tensor,   # (B, N, 2)  -- (row, col) int
        target_positions: torch.Tensor,   # (B, N, 2)  -- (row, col) int
    ) -> torch.Tensor:
        """
        Returns
        -------
        (B, N, N) doubly-stochastic assignment matrix
        assignment[b, i, j] = probability that robot i is assigned to target j
        """
        B, N = robot_masks.shape[:2]

        # 1. N encoder passes -------------------------------------------------
        F_list = []
        for i in range(N):
            x_i = torch.stack([obstacle_map, robot_masks[:, i]], dim=1)  # (B, 2, H_map, W_map)
            F_list.append(self.encoder(x_i))                              # (B, D, H_map, W_map)

        # 2. Feature extraction -----------------------------------------------
        # robot_feats[i] : feature at robot i's own position under F_i  -> (B, D)
        # edge_feats[i]  : feature at every target position under F_i   -> (B, N, D)

        robot_raw_list = []
        edge_raw_list  = []

        for i in range(N):
            pos_i = robot_positions[:, i:i+1, :]              # (B, 1, 2)
            robot_raw_list.append(
                MapEncoder.extract(F_list[i], pos_i).squeeze(1)  # (B, D)
            )
            edge_raw_list.append(
                MapEncoder.extract(F_list[i], target_positions)  # (B, N, D)
            )

        robot_raw = torch.stack(robot_raw_list, dim=1)   # (B, N, D)
        edge_raw  = torch.stack(edge_raw_list,  dim=1)   # (B, N, N, D)  [b, i, j]
        target_raw = edge_raw.mean(dim=1)                # (B, N, D)  mean over robots

        # 3. Project to GNN hidden dim ----------------------------------------
        robot_h  = self.feat_proj(robot_raw)              # (B, N, H)
        target_h = self.feat_proj(target_raw)             # (B, N, H)
        edge_h   = self.feat_proj(edge_raw)               # (B, N, N, H)

        # 4. Bipartite GNN ----------------------------------------------------
        for layer in self.gnn:
            robot_h, target_h = layer(robot_h, target_h, edge_h)

        # 5. Score matrix -----------------------------------------------------
        r_exp = robot_h.unsqueeze(2).expand(B, N, N, -1)   # (B, N, N, H)
        t_exp = target_h.unsqueeze(1).expand(B, N, N, -1)  # (B, N, N, H)
        scores = self.score_mlp(
            torch.cat([r_exp, t_exp], dim=-1)               # (B, N, N, 2H)
        ).squeeze(-1)                                        # (B, N, N)

        # 6. Sinkhorn ---------------------------------------------------------
        return sinkhorn(scores, self.sinkhorn_iters), scores


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _load_encoder(checkpoint_path: str) -> tuple[MapEncoder, int]:
    """
    Load a MapEncoder from a DistanceFieldNet checkpoint.

    Returns (encoder, enc_dim) where enc_dim = encoder output channels (D).
    Loaded on CPU; caller moves the parent model to the target device.
    """
    from src.net.map_encoder.train import load_checkpoint as _load_me
    model, _, me_cfg = _load_me(checkpoint_path, torch.device("cpu"))
    return model.encoder, me_cfg.encoder.out_channels


def build_model(cfg: BNavAssignConfig) -> NavAssignNet:
    encoder, enc_dim = _load_encoder(cfg.model.encoder_checkpoint)
    return NavAssignNet(encoder, enc_dim, cfg.model)
