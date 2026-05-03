from dataclasses import dataclass, field
from typing import Union


@dataclass
class MapConfig:
    width: int = 20
    height: int = 20


@dataclass
class ModelConfig:
    # Path to a trained MapEncoder checkpoint (DistanceFieldNet .pt file).
    # The encoder architecture is read from the checkpoint config.
    encoder_checkpoint: str = r"C:\Personal\_Faculta\_Doctorat\PhD-MAGNET\src\net\map_encoder\checkpoints\run_12_epoch_1000.pt"

    # GNN
    gnn_hidden_dim: int = 128
    gnn_layers: int = 3
    dropout: float = 0.1

    # Sinkhorn
    sinkhorn_iters: int = 20


@dataclass
class TrainConfig:
    num_robots: int = 4             # fixed N for this training run
    num_train: int = 50_000
    num_val: int = 5_000
    train_seed: int = 0             # fixed seed for reproducible training data
    batch_size: int = 32
    lr: float = 1e-3
    lr_min: float = 1e-5
    epochs: int = 1000
    encoder_freeze_epochs: int = 400  # unfreeze at the specified epoch
    val_every: int = 10
    checkpoint_every: int = 50
    device: str = "cuda"
    grad_clip: float = 1.0
    weight_decay: float = 1e-4


@dataclass
class OverfitTrainConfig:
    """Small dataset, no regularisation — use to verify the model can memorise samples."""
    num_robots: int = 4
    num_train: int = 64
    num_val: int = 16
    train_seed: int = 0
    batch_size: int = 16
    lr: float = 1e-3
    lr_min: float = 1e-5
    epochs: int = 1_000
    encoder_freeze_epochs: int = 0  # unfreeze everything immediately
    val_every: int = 10
    checkpoint_every: int = 200
    device: str = "cuda"
    grad_clip: float = 1.0
    weight_decay: float = 0.0       # no regularisation


@dataclass
class BNavAssignConfig:
    map: MapConfig = field(default_factory=MapConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: Union[TrainConfig, OverfitTrainConfig] = field(default_factory=TrainConfig)
