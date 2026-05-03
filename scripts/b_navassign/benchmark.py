"""
Timing benchmark: brute-force (N!) vs LBAP vs NavAssignNet.

Called via:  run.ps1 b_navassign --benchmark --checkpoint <path>

Brute-force and LBAP run sequentially on CPU.
Neural pipeline runs on GPU (default) or CPU (--cpu flag), with a sweep
over batch sizes to find the peak throughput.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import numpy as np
import torch

from src.net.b_navassign.dataset import (
    _bfs_distances, generate_scenario,
    bottleneck_assignment, lbap_assignment,
)
from src.net.b_navassign.train import load_checkpoint

CHECK = False

# ---------------------------------------------------------------------------
# Analytical pipelines
# ---------------------------------------------------------------------------

def _build_cost_matrix(grid, robot_positions, target_positions):
    N = len(robot_positions)
    cost_matrix = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        dist = _bfs_distances(grid, (int(robot_positions[i, 0]), int(robot_positions[i, 1])))
        for j in range(N):
            cost_matrix[i, j] = float(dist[int(target_positions[j, 0]), int(target_positions[j, 1])])
    return cost_matrix


def _bruteforce_pipeline(grid, robot_positions, target_positions):
    bottleneck_assignment(_build_cost_matrix(grid, robot_positions, target_positions))


def _lbap_pipeline(grid, robot_positions, target_positions):
    lbap_assignment(_build_cost_matrix(grid, robot_positions, target_positions))


# ---------------------------------------------------------------------------
# OM algorithm (official implementation, Aakash & Saha, AI 347, 2025)
# ---------------------------------------------------------------------------

from scripts.b_navassign.om.omotc import om_heuristic as _om_heuristic


def _om_assignment(
    grid: np.ndarray,
    robot_positions: np.ndarray,
    target_positions: np.ndarray,
) -> tuple:
    """
    OM (Optimal Makespan) goal assignment -- Algorithm 1 from:
    "A scalable multi-robot goal assignment algorithm for minimizing
    mission time followed by total movement cost"
    Aakash & Saha, Artificial Intelligence 347 (2025).

    Wraps the official reference implementation.  Returns (row_ind, col_ind).
    """
    N = len(robot_positions)
    all_start_loc = {f'r{i}': (int(robot_positions[i, 0]), int(robot_positions[i, 1]))
                     for i in range(N)}
    all_goal_loc  = {f'g{j}': (int(target_positions[j, 0]), int(target_positions[j, 1]))
                     for j in range(N)}
    result_h, *_ = _om_heuristic(N, N, grid, all_start_loc, all_goal_loc, '2D')
    col_ind = np.empty(N, dtype=int)
    for (robot_name, goal_name), _ in result_h:
        col_ind[int(robot_name[1:])] = int(goal_name[1:])
    return np.arange(N), col_ind


def _om_pipeline(grid, robot_positions, target_positions):
    """
    OM -- state-of-the-art scalable makespan assignment.
    Aakash & Saha, Artificial Intelligence 347 (2025).
    """
    _om_assignment(grid, robot_positions, target_positions)


# ---------------------------------------------------------------------------
# Neural pipeline
# ---------------------------------------------------------------------------

def _neural_assignment(model, obs, masks, r_pos, t_pos):
    with torch.no_grad():
        model(obs, masks, r_pos, t_pos)


def _make_tensors(batch, device):
    obs   = torch.stack([torch.tensor(s["obstacle_map"])                       for s in batch]).to(device)
    masks = torch.stack([torch.tensor(s["robot_masks"])                        for s in batch]).to(device)
    r_pos = torch.stack([torch.tensor(s["robot_positions"], dtype=torch.long)  for s in batch]).to(device)
    t_pos = torch.stack([torch.tensor(s["target_positions"], dtype=torch.long) for s in batch]).to(device)
    return obs, masks, r_pos, t_pos


def _sync(use_gpu):
    if use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def run_check(cfg, n: int = 500, seed: int = 77) -> None:
    """
    Run all three analytical solvers on the same scenario pool and verify
    that LBAP and OM reach the same optimal makespan as brute-force.

    Since generate_scenario discards tied scenarios, the optimal assignment
    is always unique -- so the achieved makespan must be identical across
    all three solvers.  Any discrepancy is a bug.
    """
    print(f"Correctness check — {cfg.train.num_robots} robots, "
          f"{cfg.map.width}x{cfg.map.height} map, {n} scenarios\n")

    rng = np.random.default_rng(seed)
    scenarios, grids = [], []
    while len(scenarios) < n:
        s = generate_scenario(cfg.map.width, cfg.map.height, cfg.train.num_robots, rng)
        if s is not None:
            scenarios.append(s)
            grids.append((s["obstacle_map"] > 0).astype(np.int8))

    lbap_fail = om_fail = 0

    for idx, (s, grid) in enumerate(zip(scenarios, grids)):
        rp, tp = s["robot_positions"], s["target_positions"]
        cm     = s["cost_matrix"]   # precomputed BFS costs

        _, bf_col  = bottleneck_assignment(cm)
        _, lbap_col = lbap_assignment(cm)
        _, om_col   = _om_assignment(grid, rp, tp)

        N = len(bf_col)
        bf_ms   = max(cm[i, bf_col[i]]   for i in range(N))
        lbap_ms = max(cm[i, lbap_col[i]] for i in range(N))
        om_ms   = max(cm[i, om_col[i]]   for i in range(N))

        if lbap_ms != bf_ms:
            lbap_fail += 1
            print(f"  [#{idx}] LBAP mismatch: bf_ms={bf_ms:.1f}  lbap_ms={lbap_ms:.1f}  "
                  f"bf={bf_col.tolist()}  lbap={lbap_col.tolist()}")

        if om_ms != bf_ms:
            om_fail += 1
            print(f"  [#{idx}] OM   mismatch: bf_ms={bf_ms:.1f}  om_ms={om_ms:.1f}  "
                  f"bf={bf_col.tolist()}  om={om_col.tolist()}")
            # Print the scenario for debugging
            print(f"    Robot positions: {rp.tolist()}")
            print(f"    Target positions: {tp.tolist()}")
            # print(f"    Obstacle map:\n{s['obstacle_map']}")
            print("    Obstacle Map:")
            for r in range(s["obstacle_map"].shape[0]):
                row_str = ",".join(str(int(x)) for x in s["obstacle_map"][r])
                print(f"    [{row_str}],")

    print(f"\nResults ({n} scenarios):")
    print(f"  LBAP failures : {lbap_fail}")
    print(f"  OM   failures : {om_fail}")
    if lbap_fail == 0 and om_fail == 0:
        print("  All assignments match brute-force. ✓")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    checkpoint:  str,
    n:           int   = 10_000,
    batch_sizes: list  = None,
    seed:        int   = 77,
    use_gpu:     bool  = True,
):
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024]

    use_gpu = use_gpu and torch.cuda.is_available()
    device  = torch.device("cuda" if use_gpu else "cpu")
    hw      = "GPU" if use_gpu else "CPU"

    model, epoch, cfg = load_checkpoint(checkpoint, device)

    if CHECK:
        run_check(cfg, n=min(n, 500), seed=seed)
        return
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Benchmarking on {hw} — {cfg.train.num_robots} robots, "
          f"{cfg.map.width}x{cfg.map.height} map, pool={n} scenarios\n")

    # --- Generate fixed scenario pool ---
    print("Generating scenario pool...")
    rng = np.random.default_rng(seed)
    scenarios = []
    while len(scenarios) < n:
        s = generate_scenario(cfg.map.width, cfg.map.height, cfg.train.num_robots, rng)
        if s is not None:
            scenarios.append(s)
    print(f"  Done — {n} scenarios\n")

    grids = [(s["obstacle_map"] > 0).astype(np.int8) for s in scenarios]

    # --- Brute-force ---
    print("Running brute-force (N! enumeration)...")
    t0 = time.perf_counter()
    for s, grid in zip(scenarios, grids):
        _bruteforce_pipeline(grid, s["robot_positions"], s["target_positions"])
    t_brute = time.perf_counter() - t0
    print(f"  total {t_brute * 1000:.1f}ms | per scenario {t_brute / n * 1000:.3f}ms\n")

    # --- LBAP ---
    print("Running LBAP (binary search + bipartite matching)...")
    t0 = time.perf_counter()
    for s, grid in zip(scenarios, grids):
        _lbap_pipeline(grid, s["robot_positions"], s["target_positions"])
    t_lbap = time.perf_counter() - t0
    print(f"  total {t_lbap * 1000:.1f}ms | per scenario {t_lbap / n * 1000:.3f}ms\n")

    # --- OM ---
    print("Running OM (lazy BFS + threshold matching, Aakash & Saha 2025)...")
    t0 = time.perf_counter()
    for s, grid in zip(scenarios, grids):
        _om_pipeline(grid, s["robot_positions"], s["target_positions"])
    t_om = time.perf_counter() - t0
    print(f"  total {t_om * 1000:.1f}ms | per scenario {t_om / n * 1000:.3f}ms\n")

    # --- Neural sweep ---
    warmup_obs, warmup_masks, warmup_rpos, warmup_tpos = _make_tensors(scenarios[:1], device)
    _neural_assignment(model, warmup_obs, warmup_masks, warmup_rpos, warmup_tpos)
    _sync(use_gpu)

    print(f"Neural pipeline ({hw}):")
    print(f"  Brute-force baseline: {t_brute / n * 1000:.3f}ms/scenario")
    print(f"  LBAP baseline:        {t_lbap  / n * 1000:.3f}ms/scenario")
    print(f"  OM baseline:          {t_om    / n * 1000:.3f}ms/scenario")
    print(f"\n{'Batch':>8}  {'Neural (total)':>16}  {'Neural (per)':>14}  "
          f"{'vs Brute':>10}  {'vs LBAP':>9}  {'vs OM':>7}")
    print("-" * 75)

    for bs in sorted(batch_sizes):
        t_neural = 0.0
        for i in range(0, n, bs):
            batch = scenarios[i:i + bs]
            obs, masks, r_pos, t_pos = _make_tensors(batch, device)
            _sync(use_gpu)
            t0 = time.perf_counter()
            _neural_assignment(model, obs, masks, r_pos, t_pos)
            _sync(use_gpu)
            t_neural += time.perf_counter() - t0

        neural_per = t_neural / n * 1000
        print(f"{bs:>8}  "
              f"{t_neural * 1000:>14.1f}ms  "
              f"{neural_per:>12.3f}ms  "
              f"{t_brute / t_neural:>9.2f}x  "
              f"{t_lbap  / t_neural:>8.2f}x  "
              f"{t_om    / t_neural:>6.2f}x")
