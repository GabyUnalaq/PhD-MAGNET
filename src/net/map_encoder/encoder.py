"""
MapEncoder: fully-convolutional spatial feature extractor.

Input  : (B, in_channels, H, W)
Output : (B, out_channels, H, W)  — same spatial resolution as input

Spatial resolution is preserved throughout (no pooling, no stride).
Dilated residual blocks expand the receptive field without downsampling:

    n_dilated_layers = 4  ->  dilations 1, 2, 4, 8
    receptive field  = 2 x (1+2+4+8) + 1 = 31 x 31

This covers the full 20x20 map so every output cell encodes global context.

After pre-training, extract a D-dim feature for any grid position (r, c) by:
    f = encoder(x)[:, :, r, c]   # (B, D)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedResBlock(nn.Module):
    """Single dilated conv with a residual connection. Shape: (B, C, H, W) -> (B, C, H, W)."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=dilation, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.bn(self.conv(x)), inplace=True)


class MapEncoder(nn.Module):
    """
    Parameters
    ----------
    in_channels      : number of input channels (default 2: obstacle map + source mask)
    base_channels    : channel width used inside all dilated blocks
    out_channels     : D, the output feature dimension per spatial cell
    n_dilated_layers : number of dilated blocks; dilation of block i = 2^i
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        out_channels: int = 64,
        n_dilated_layers: int = 4,
    ):
        super().__init__()

        # Project input channels -> base_channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stack of dilated residual blocks: dilations 1, 2, 4, 8, ...
        self.blocks = nn.ModuleList([
            _DilatedResBlock(base_channels, dilation=2 ** i)
            for i in range(n_dilated_layers)
        ])

        # Project base_channels -> out_channels (no activation — raw features)
        self.output_proj = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        (B, out_channels, H, W) — spatial feature map at full input resolution
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)

    @staticmethod
    def extract(F_map: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Index a pre-computed feature map at a set of grid positions.
        Call forward() once, then call this as many times as needed.

        Parameters
        ----------
        F_map     : (B, D, H, W) — output of forward()
        positions : (B, N, 2)    — integer (row, col) positions to sample

        Returns
        -------
        (B, N, D) — feature vector at each position
        """
        B, D, H, W = F_map.shape
        N = positions.shape[1]
        rows = positions[:, :, 0].long().clamp(0, H - 1)           # (B, N)
        cols = positions[:, :, 1].long().clamp(0, W - 1)           # (B, N)
        idx = (rows * W + cols).unsqueeze(1).expand(B, D, N)       # (B, D, N)
        flat = F_map.view(B, D, H * W)                              # (B, D, H*W)
        return flat.gather(2, idx).permute(0, 2, 1)                 # (B, N, D)
