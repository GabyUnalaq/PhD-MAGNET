from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    in_channels: int = 2        # obstacle map + source mask
    base_channels: int = 32     # internal channel width throughout the network
    out_channels: int = 64      # D — feature dimension per spatial cell in the output
    n_dilated_layers: int = 4   # dilations: 1, 2, 4, 8 -> receptive field 31x31 on a 20x20 map


@dataclass
class TrainConfig:
    map_width: int = 20
    map_height: int = 20
    num_train: int = 50_000     # number of (map, source) training pairs
    num_val: int = 5_000
    batch_size: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-6
    epochs: int = 1000
    val_every: int = 5
    checkpoint_every: int = 100
    device: str = "cuda"
    grad_clip: float = 1.0


@dataclass
class MapEncoderConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
