import math
import numpy as np

from src.map._grid_map import GridMap

_DEFAULT_N_CANDIDATES = 20


# ---------------------------------------------------------------------------
# Core: candidate point generation
# ---------------------------------------------------------------------------

def generate_candidate_points(
    grid_map: GridMap,
    n: int = _DEFAULT_N_CANDIDATES,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate n well-distributed free positions on the map using farthest-point
    sampling. Each position is a cell center: (gx = row + 0.5, gy = col + 0.5).

    Returns
    -------
    np.ndarray of shape (n, 2), dtype float32, columns [gx, gy].
    """
    rng = np.random.default_rng(seed)

    free_rows, free_cols = np.where(grid_map.grid == 0)
    if len(free_rows) == 0:
        raise ValueError("No free cells on the map.")
    if n > len(free_rows):
        raise ValueError(f"Requested {n} points but only {len(free_rows)} free cells exist.")

    pool = np.stack([free_rows + 0.5, free_cols + 0.5], axis=1).astype(np.float32)

    # Farthest-point sampling for good spatial spread
    chosen_idx = [int(rng.integers(len(pool)))]
    min_dists = np.full(len(pool), np.inf)

    while len(chosen_idx) < n:
        last = pool[chosen_idx[-1]]
        dists = np.sum((pool - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        chosen_idx.append(int(np.argmax(min_dists)))

    return pool[chosen_idx]


# ---------------------------------------------------------------------------
# Problem formulations
# ---------------------------------------------------------------------------

def make_linked_pairs(
    grid_map: GridMap,
    points: np.ndarray,
    num_pairs_range: tuple[int, int],
    min_travel_distance: float | None = None,
    seed: int | None = None,
) -> GridMap:
    """
    Assign start/finish pairs to grid_map.linked_points from a candidate pool.

    Each pair is drawn from `points` without replacement. Start and finish must
    be at least `min_travel_distance` apart (defaults to half the map diagonal).

    Parameters
    ----------
    grid_map : GridMap
    points : (N, 2) array of candidate positions [gx, gy]
    num_pairs_range : (min, max) number of pairs, inclusive
    min_travel_distance : minimum Euclidean distance per pair
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    width, height = grid_map.size

    if min_travel_distance is None:
        min_travel_distance = 0.5 * math.sqrt(width ** 2 + height ** 2)

    num_pairs = int(rng.integers(num_pairs_range[0], num_pairs_range[1] + 1))
    if num_pairs * 2 > len(points):
        raise ValueError(
            f"Need {num_pairs * 2} points for {num_pairs} pairs but only {len(points)} candidates available."
        )

    shuffled = points[rng.permutation(len(points))]
    used = set()
    linked: list[list[float]] = []

    for pair_idx in range(num_pairs):
        placed = False
        for start in shuffled:
            sk = (start[0], start[1])
            if sk in used:
                continue
            for finish in shuffled:
                fk = (finish[0], finish[1])
                if fk in used or fk == sk:
                    continue
                dist = math.sqrt((start[0] - finish[0]) ** 2 + (start[1] - finish[1]) ** 2)
                if dist >= min_travel_distance:
                    used.add(sk)
                    used.add(fk)
                    linked.append([start[0], start[1], finish[0], finish[1]])
                    placed = True
                    break
            if placed:
                break

        if not placed:
            raise ValueError(
                f"Could not place pair {pair_idx + 1}: min_travel_distance "
                f"({min_travel_distance:.1f}) may be too strict for the available candidates."
            )

    grid_map.linked_points = np.array(linked, dtype=np.float32)
    return grid_map


def make_start_points(
    grid_map: GridMap,
    points: np.ndarray,
    n: int | None = None,
    seed: int | None = None,
) -> GridMap:
    """
    Assign positions to grid_map.start_points.

    Parameters
    ----------
    grid_map : GridMap
    points : (N, 2) candidate positions [gx, gy]
    n : number of start points to pick (None = use all)
    seed : random seed (only used when n < len(points))
    """
    if n is None or n == len(points):
        selected = points
    else:
        if n > len(points):
            raise ValueError(f"Requested {n} start points but only {len(points)} candidates available.")
        rng = np.random.default_rng(seed)
        selected = points[rng.choice(len(points), size=n, replace=False)]

    grid_map.start_points = np.array(selected, dtype=np.float32)
    return grid_map


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def populate_grid_map(
    grid_map: GridMap,
    num_robots_range: tuple[int, int],
    n_candidates: int = _DEFAULT_N_CANDIDATES,
    min_travel_distance: float | None = None,
    seed: int | None = None,
) -> GridMap:
    """
    End-to-end convenience: generate candidates then assign linked pairs.

    Equivalent to calling generate_candidate_points -> make_linked_pairs.
    """
    rng = np.random.default_rng(seed)
    s1, s2 = int(rng.integers(0, 99999, size=2)[0]), int(rng.integers(0, 99999, size=2)[1])

    points = generate_candidate_points(grid_map, n=n_candidates, seed=s1)
    return make_linked_pairs(grid_map, points, num_robots_range, min_travel_distance, seed=s2)
