"""
Training dataset for MapEncoder.

Each sample is generated from scratch:
  1. Random obstacle map via generate_structured_grid_map (fully connected, bordered)
  2. One random free cell chosen as the source
  3. BFS from source -> distance field, normalised to [0, 1]

Returned tensors:
  input : (2, H, W) float32  — channel 0: obstacle map, channel 1: source mask
  target: (1, H, W) float32  — normalised proximity field (1 at source, 0 at farthest cell)
  mask  : (1, H, W) float32  — 1 at free cells, 0 at obstacles (used to mask the loss)

All free cells are guaranteed reachable (generate_structured_grid_map ensures full connectivity).
"""
import numpy as np
import torch
from collections import deque
from torch.utils.data import Dataset

from src.map.generator.grid_map_generator import generate_structured_grid_map


# ---------------------------------------------------------------------------
# Core sample generation
# ---------------------------------------------------------------------------

def _bfs_distance_field(grid: np.ndarray, source_rc: tuple[int, int]) -> np.ndarray:
    """
    BFS distance from source_rc to every reachable free cell.

    Parameters
    ----------
    grid      : (H, W) int8 — 1 = obstacle, 0 = free
    source_rc : (row, col) of the source cell

    Returns
    -------
    (H, W) float32 — proximity field normalised to [0, 1]; source = 1,
    farthest reachable cell = 0, obstacle cells = 0
    """
    H, W = grid.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    sr, sc = source_rc
    dist[sr, sc] = 0
    queue = deque([(sr, sc)])

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    max_dist = int(dist.max())
    field = np.where(dist >= 0, 1.0 - dist / max(max_dist, 1), 0.0).astype(np.float32)
    return field


def generate_sample(
    width: int,
    height: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one (obstacle_map, source_mask, distance_field) triple.

    Parameters
    ----------
    width, height : map dimensions
    seed          : full reproducibility when set

    Returns
    -------
    obs      : (H, W) float32 — obstacle map (1 = obstacle, 0 = free)
    src_mask : (H, W) float32 — 1 at source cell, 0 elsewhere
    dist     : (H, W) float32 — proximity field normalised to [0, 1] (1 at source, 0 at farthest)
    """
    rng = np.random.default_rng(seed)
    map_seed, src_seed = rng.integers(0, 99999, size=2).tolist()

    grid_map = generate_structured_grid_map(width=width, height=height, seed=map_seed)
    grid = grid_map.grid   # (H, W) int8

    free_cells = np.argwhere(grid == 0)
    rng_src = np.random.default_rng(src_seed)
    src_rc = tuple(free_cells[rng_src.integers(len(free_cells))].tolist())

    src_mask = np.zeros((height, width), dtype=np.float32)
    src_mask[src_rc] = 1.0

    dist = _bfs_distance_field(grid, src_rc)

    return grid.astype(np.float32), src_mask, dist


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MapEncoderDataset(Dataset):
    """
    Generates samples on construction and holds them in memory.

    Each item: (input (2,H,W), target (1,H,W), mask (1,H,W))
    """

    def __init__(
        self,
        n: int,
        width: int,
        height: int,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        while len(self.samples) < n:
            s = int(rng.integers(0, 99999))
            try:
                obs, src, dist = generate_sample(width, height, seed=s)
                inp    = torch.tensor(np.stack([obs, src], axis=0), dtype=torch.float)  # (2,H,W)
                target = torch.tensor(dist,               dtype=torch.float).unsqueeze(0)  # (1,H,W)
                mask   = torch.tensor((obs == 0).astype(np.float32)).unsqueeze(0)          # (1,H,W)
                self.samples.append((inp, target, mask))
            except (ValueError, RuntimeError):
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]
