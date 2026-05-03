import numpy as np
from noise import pnoise2
from scipy.ndimage import generic_filter, label
from scipy.spatial import cKDTree
from collections import deque

from src.map._grid_map import GridMap
from src.map._base_map import MapType


def _perlin_noise_grid(width: int, height: int, scale: float, octaves: int, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ox, oy = rng.integers(0, 10000, size=2) if seed is not None else (0, 0)

    xs = (np.arange(width) + ox) / scale
    ys = (np.arange(height) + oy) / scale

    grid = np.array([
        [pnoise2(float(xs[x]), float(ys[y]), octaves=octaves, persistence=0.5, lacunarity=2.0)
         for x in range(width)]
        for y in range(height)
    ])
    return grid


def _smooth_cellular_automata(binary_map: np.ndarray, iterations: int, birth_limit: int, death_limit: int) -> np.ndarray:
    def _step(grid: np.ndarray) -> np.ndarray:
        neighbor_sum = generic_filter(grid.astype(float), np.sum, size=3, mode='constant', cval=1.0)
        neighbor_sum -= grid  # exclude cell itself

        new_grid = grid.copy()
        new_grid[(grid == 1) & (neighbor_sum <= death_limit)] = 0
        new_grid[(grid == 0) & (neighbor_sum >= birth_limit)] = 1
        return new_grid

    grid = binary_map.copy()
    for _ in range(iterations):
        grid = _step(grid)
    return grid


def _add_border_obstacles(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()
    result[0, :] = 1
    result[-1, :] = 1
    result[:, 0] = 1
    result[:, -1] = 1
    return result


def _min_obstacle_path(grid: np.ndarray, start: tuple, end: tuple) -> list[tuple]:
    """0-1 BFS from start to end; free cells cost 0, obstacle cells cost 1."""
    h, w = grid.shape
    dist = np.full((h, w), np.inf)
    dist[start] = 0
    prev = {start: None}
    dq = deque([start])

    while dq:
        r, c = dq.popleft()
        if (r, c) == end:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cost = int(grid[nr, nc])
                new_dist = dist[r, c] + cost
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    prev[(nr, nc)] = (r, c)
                    if cost == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))

    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return path


def _connect_regions(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()

    while True:
        labeled, num_regions = label(result == 0)
        if num_regions <= 1:
            break

        region1_cells = np.argwhere(labeled == 1)
        other_cells = np.argwhere((labeled > 1))

        _, idxs = cKDTree(other_cells).query(region1_cells)
        dists = np.linalg.norm(other_cells[idxs] - region1_cells, axis=1)
        closest = np.argmin(dists)

        start = tuple(region1_cells[closest])
        end = tuple(other_cells[idxs[closest]])

        for cell in _min_obstacle_path(result, start, end):
            result[cell] = 0

    return result


def generate_structured_grid_map(
    width: int,
    height: int,
    scale: float = 5.0,
    octaves: int = 3,
    threshold: float = 0.05,
    birth_limit: int = 4,
    death_limit: int = 3,
    seed: int | None = None,
) -> GridMap:
    iterations = max(2, round(max(width, height) / 12))

    noise_grid = _perlin_noise_grid(width, height, scale, octaves, seed)
    binary_map = (noise_grid > threshold).astype(int)
    smoothed = _smooth_cellular_automata(binary_map, iterations, birth_limit, death_limit)
    bordered = _add_border_obstacles(smoothed)
    connected = _connect_regions(bordered)

    grid_map = GridMap(MapType.MAP, width, height)
    grid_map.grid = connected.astype(np.int8)
    return grid_map

def generate_random_grid_map(
        width: int,
        height: int,
        obstacle_prob: float = 0.1,
        seed: int | None = None,
) -> GridMap:
    rng = np.random.default_rng(seed)
    grid = (rng.random((height, width)) < obstacle_prob).astype(np.int8)
    bordered = _add_border_obstacles(grid)
    connected = _connect_regions(bordered)

    grid_map = GridMap(MapType.MAP, width, height)
    grid_map.grid = connected
    return grid_map