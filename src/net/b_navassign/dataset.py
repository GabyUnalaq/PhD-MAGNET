"""
Training dataset for BNavAssign (Bottleneck NavAssign).

Each sample is a robot-to-target assignment scenario:
  1. Random obstacle map (fully connected, bordered)
  2. N random free cells as robot start positions
  3. N random free cells as target positions
  4. N x N BFS cost matrix (BFS = A* on uniform grid, exact)
  5. Bottleneck-optimal assignment: the permutation minimising the MAXIMUM
     individual travel cost (makespan). All robots move simultaneously, so
     the last robot to arrive determines task completion time.

Trivial scenarios (greedy nearest-target makespan == bottleneck optimal
makespan) are kept at a controlled fraction (default 10%) so the model
sees challenging cases.

Returned tensors per sample (from __getitem__):
  obstacle_map     : (H, W)    float32 -- 1 = obstacle, 0 = free
  robot_masks      : (N, H, W) float32 -- robot_masks[i] has a 1 at robot i's position
  robot_positions  : (N, 2)    int64   -- (row, col) for each robot
  target_positions : (N, 2)    int64   -- (row, col) for each target
  assignment       : (N, N)    float32 -- ground-truth permutation matrix
"""
import numpy as np
import torch
from collections import deque
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from src.map.generator.grid_map_generator import generate_structured_grid_map


# ---------------------------------------------------------------------------
# BFS cost
# ---------------------------------------------------------------------------

def _bfs_distances(grid: np.ndarray, source_rc: tuple[int, int]) -> np.ndarray:
    """
    BFS shortest distances from source_rc to all reachable free cells.

    Parameters
    ----------
    grid      : (H, W) int8 -- 1 = obstacle, 0 = free
    source_rc : (row, col) of the source cell

    Returns
    -------
    (H, W) int32 -- distance in steps; -1 for obstacles and unreachable cells
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
    return dist


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def bottleneck_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Brute-force bottleneck assignment: find the permutation minimising the
    maximum individual travel cost (makespan).

    Enumerates all N! permutations and picks the one with the smallest
    max(cost_matrix[i, perm[i]] for i in range(N)).

    Returns (row_ind, col_ind) if the optimum is unique, or None if two or
    more permutations share the same minimum max cost (tied GT -- discard).

    This function is intentionally isolated so it can be swapped for a
    state-of-the-art bottleneck assignment solver without touching the rest
    of the dataset pipeline.
    """
    N = len(cost_matrix)
    best_cost = float("inf")
    best_perm = None
    tied      = False

    for perm in permutations(range(N)):
        max_cost = max(cost_matrix[i, perm[i]] for i in range(N))
        if max_cost < best_cost:
            best_cost = max_cost
            best_perm = perm
            tied      = False
        elif max_cost == best_cost:
            tied = True

    if tied:
        return None

    return np.arange(N), np.array(best_perm)


def lbap_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Linear Bottleneck Assignment Problem (LBAP) solver.

    Finds the assignment minimising the maximum individual cost without
    enumerating all N! permutations.  Algorithm:
      1. Collect all unique cost values and sort them ascending.
      2. Binary search for the minimum threshold T such that the bipartite
         graph G_T = {(i,j) | cost[i,j] <= T} has a perfect matching.
      3. Extract a perfect matching from G_T using the Hungarian algorithm
         (linear_sum_assignment with a 0/1 adjacency mask).

    Time complexity: O(N^2 log N) vs O(N! * N) for brute-force.

    Returns (row_ind, col_ind) if the optimum is unique, or None if two or
    more permutations share the same minimum max cost (tied GT -- discard).
    Tie check is performed by verifying no other threshold yields a valid
    perfect matching with a strictly equal maximum cost via a different
    matching.
    """
    N = len(cost_matrix)
    thresholds = np.unique(cost_matrix)

    # Binary search: find smallest T with a perfect matching in G_T
    lo, hi = 0, len(thresholds) - 1
    best_T_idx = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        T = thresholds[mid]
        mask = (cost_matrix <= T).astype(np.float32)
        # Use Hungarian on negated mask: maximise number of covered pairs
        row_ind, col_ind = linear_sum_assignment(-mask)
        if mask[row_ind, col_ind].sum() == N:   # perfect matching found
            best_T_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    T_opt = thresholds[best_T_idx]
    mask  = (cost_matrix <= T_opt).astype(np.float32)
    row_ind, col_ind = linear_sum_assignment(-mask)

    # Tie check: count distinct perfect matchings at this threshold
    # A tie exists if removing any matched edge still leaves a perfect matching
    # in G_T, implying an alternative optimal assignment.
    # Efficient check: try all N candidate edge removals.
    tied = False
    for k in range(N):
        reduced = mask.copy()
        reduced[row_ind[k], col_ind[k]] = 0.0
        r2, c2 = linear_sum_assignment(-reduced)
        if reduced[r2, c2].sum() == N:
            tied = True
            break

    if tied:
        return None

    return row_ind, col_ind


def _is_trivial(cost_matrix: np.ndarray, bottleneck_col: np.ndarray) -> bool:
    """
    True if the greedy assignment (each robot independently picks its nearest
    target) achieves the same makespan as the bottleneck-optimal assignment.

    If greedy produces conflicts (two robots want the same target), the
    scenario is NOT trivially solvable by nearest-neighbour.
    """
    greedy = np.argmin(cost_matrix, axis=1)
    if len(set(greedy.tolist())) < len(greedy):
        return False
    N = len(cost_matrix)
    greedy_max     = max(cost_matrix[i, greedy[i]]        for i in range(N))
    bottleneck_max = max(cost_matrix[i, bottleneck_col[i]] for i in range(N))
    return bool(greedy_max == bottleneck_max)


def generate_scenario(
    width: int,
    height: int,
    num_robots: int,
    rng: np.random.Generator,
) -> dict | None:
    """
    Generate one assignment scenario.

    Returns a dict with numpy arrays, or None if generation fails
    (not enough free cells -- extremely unlikely with generate_structured_grid_map).
    """
    map_seed = int(rng.integers(0, 2 ** 31))
    grid_map = generate_structured_grid_map(width=width, height=height, seed=map_seed)
    grid = grid_map.grid  # (H, W) int8

    free_cells = np.argwhere(grid == 0)
    if len(free_cells) < 2 * num_robots:
        return None

    # Sample 2*N distinct free cells
    indices = rng.choice(len(free_cells), size=2 * num_robots, replace=False)
    robot_positions = free_cells[indices[:num_robots]]   # (N, 2)
    target_positions = free_cells[indices[num_robots:]]  # (N, 2)

    # N x N cost matrix: one BFS per robot
    cost_matrix = np.zeros((num_robots, num_robots), dtype=np.float32)
    for i, (rr, rc) in enumerate(robot_positions):
        dist = _bfs_distances(grid, (int(rr), int(rc)))
        for j, (tr, tc) in enumerate(target_positions):
            d = dist[int(tr), int(tc)]
            if d < 0:
                return None  # unreachable -- impossible with connected map, guard only
            cost_matrix[i, j] = float(d)

    # Bottleneck-optimal assignment (brute-force over N! permutations)
    result = bottleneck_assignment(cost_matrix)
    if result is None:
        return None  # tied optimal -- discard to avoid ambiguous GT
    row_ind, col_ind = result
    # col_ind[i] = index of target assigned to robot i

    trivial = _is_trivial(cost_matrix, col_ind)

    # Permutation matrix: assignment[i, col_ind[i]] = 1
    assignment = np.zeros((num_robots, num_robots), dtype=np.float32)
    assignment[row_ind, col_ind] = 1.0

    # Robot masks
    robot_masks = np.zeros((num_robots, height, width), dtype=np.float32)
    for i, (rr, rc) in enumerate(robot_positions):
        robot_masks[i, int(rr), int(rc)] = 1.0

    return {
        "obstacle_map":     grid.astype(np.float32),
        "robot_masks":      robot_masks,
        "robot_positions":  robot_positions.astype(np.int64),
        "target_positions": target_positions.astype(np.int64),
        "assignment":       assignment,
        "cost_matrix":      cost_matrix,             # (N, N) float32 -- BFS distances
        "trivial":          trivial,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NavAssignDataset(Dataset):
    """
    Generates assignment scenarios on construction and holds them in memory.

    trivial_fraction controls what fraction of samples are trivial (greedy ==
    Hungarian). The rest are non-trivial (challenging) scenarios.

    Each item: (obstacle_map, robot_masks, robot_positions, target_positions, assignment)
    """

    def __init__(
        self,
        n: int,
        width: int,
        height: int,
        num_robots: int,
        trivial_fraction: float = 0.1,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)

        n_trivial_target    = max(1, int(round(n * trivial_fraction)))
        n_nontrivial_target = n - n_trivial_target

        trivial_pool:    list[dict] = []
        nontrivial_pool: list[dict] = []

        while len(nontrivial_pool) < n_nontrivial_target or len(trivial_pool) < n_trivial_target:
            s = generate_scenario(width, height, num_robots, rng)
            if s is None:
                continue
            if s["trivial"] and len(trivial_pool) < n_trivial_target:
                trivial_pool.append(s)
            elif not s["trivial"] and len(nontrivial_pool) < n_nontrivial_target:
                nontrivial_pool.append(s)

        combined = nontrivial_pool + trivial_pool
        order = rng.permutation(len(combined))

        self.samples: list[tuple] = []
        for idx in order:
            s = combined[idx]
            self.samples.append((
                torch.tensor(s["obstacle_map"]),                        # (H, W)
                torch.tensor(s["robot_masks"]),                         # (N, H, W)
                torch.tensor(s["robot_positions"],  dtype=torch.long),  # (N, 2)
                torch.tensor(s["target_positions"], dtype=torch.long),  # (N, 2)
                torch.tensor(s["assignment"]),                          # (N, N)
                torch.tensor(s["cost_matrix"]),                         # (N, N)
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]
