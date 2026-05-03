import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import cv2

from src.net.b_navassign.config import BNavAssignConfig
from src.net.b_navassign.dataset import generate_scenario

CELL   = 28          # pixels per grid cell
ROBOT_COLOR  = (220, 100,  30)   # BGR: blue-ish
TARGET_COLOR = ( 30, 100, 220)   # BGR: red-ish
GT_COLOR     = ( 50, 200,  50)   # BGR: green
PRED_COLOR   = ( 30, 200, 220)   # BGR: yellow

SHOW_ASSIGNMENT_LINES = True  # works with view_gt


def _render_scenario(
    obs:       np.ndarray,            # (H, W) float32 obstacle map
    robot_pos: np.ndarray,            # (N, 2) int  row,col
    tgt_pos:   np.ndarray,            # (N, 2) int  row,col
    assignment: np.ndarray | None,    # (N, N) float32 or None
    line_color: tuple,
) -> np.ndarray:
    """
    Render one assignment panel.

    obstacle cells: dark gray
    free cells    : light gray
    robots        : filled circles (ROBOT_COLOR)
    targets       : filled squares (TARGET_COLOR)
    assignment    : lines from robot centre to target centre (line_color)
    """
    H, W = obs.shape
    img = np.zeros((H * CELL, W * CELL, 3), dtype=np.uint8)

    # Background
    for r in range(H):
        for c in range(W):
            color = (40, 40, 40) if obs[r, c] == 1 else (200, 200, 200)
            cv2.rectangle(img,
                          (c * CELL,       r * CELL),
                          ((c + 1) * CELL - 1, (r + 1) * CELL - 1),
                          color, -1)

    def cell_centre(r, c):
        return (c * CELL + CELL // 2, r * CELL + CELL // 2)

    # Assignment lines (drawn under markers)
    if assignment is not None:
        hard = assignment.argmax(axis=-1)    # (N,) -- target index per robot
        for i, ti in enumerate(hard):
            cx1, cy1 = cell_centre(robot_pos[i, 0], robot_pos[i, 1])
            cx2, cy2 = cell_centre(tgt_pos[ti,   0], tgt_pos[ti,   1])
            cv2.line(img, (cx1, cy1), (cx2, cy2), line_color, 2, cv2.LINE_AA)

    # Target markers (squares)
    for ti, (tr, tc) in enumerate(tgt_pos):
        cx, cy = cell_centre(tr, tc)
        half = CELL // 2 - 2
        cv2.rectangle(img,
                      (cx - half, cy - half),
                      (cx + half, cy + half),
                      TARGET_COLOR, -1)

    # Robot markers (circles)
    for _, (rr, rc) in enumerate(robot_pos):
        cx, cy = cell_centre(rr, rc)
        cv2.circle(img, (cx, cy), CELL // 2 - 2, ROBOT_COLOR, -1)

    return img


def main():
    import argparse
    from src.net.b_navassign.train import train, evaluate, load_checkpoint

    parser = argparse.ArgumentParser(description="Bottleneck NavAssign — train / evaluate / visualise / benchmark")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train",     action="store_true", help="Train NavAssignNet")
    mode.add_argument("--test",      action="store_true", help="Evaluate a checkpoint on a held-out set")
    mode.add_argument("--visualise", action="store_true", help="Show predicted vs GT assignment (requires --checkpoint)")
    mode.add_argument("--view_gt",   action="store_true", help="Show ground-truth assignment scenarios")
    mode.add_argument("--benchmark", action="store_true", help="Time brute-force vs LBAP vs neural (requires --checkpoint)")
    parser.add_argument("--checkpoint",        type=str,  default=None, help="Path to NavAssign .pt checkpoint")
    parser.add_argument("--encoder_checkpoint",type=str,  default=None, help="Path to MapEncoder .pt checkpoint (--train)")
    parser.add_argument("--n",                 type=int,  default=3,    help="Number of scenarios to show (--view_gt/--visualise) or scenario pool size (--benchmark)")
    parser.add_argument("--seed",              type=int,  default=None, help="RNG seed for scenario generation")
    parser.add_argument("--batch_sizes",       type=int,  nargs="+", default=[1, 8, 16, 32, 64, 128, 256, 512, 1024], help="Batch sizes for --benchmark neural sweep")
    parser.add_argument("--cpu",               action="store_true", help="Force CPU for neural pipeline in --benchmark")
    args = parser.parse_args()

    if args.train:
        cfg = BNavAssignConfig()
        if args.encoder_checkpoint:
            cfg.model.encoder_checkpoint = args.encoder_checkpoint
        train(cfg)

    elif args.test:
        if not args.checkpoint:
            parser.error("--test requires --checkpoint <path>")
        evaluate(args.checkpoint)

    elif args.view_gt:
        cfg = BNavAssignConfig()
        rng = np.random.default_rng(args.seed)

        for i in range(args.n):
            while True:
                s = generate_scenario(cfg.map.width, cfg.map.height, cfg.train.num_robots, rng)
                if s is not None:
                    break

            trivial_str = "trivial" if s["trivial"] else "non-trivial"
            print(f"Scenario {i}  [{trivial_str}]  robots={s['robot_positions'].tolist()}  targets={s['target_positions'].tolist()}")

            img = _render_scenario(
                s["obstacle_map"], s["robot_positions"], s["target_positions"],
                s["assignment"] if SHOW_ASSIGNMENT_LINES else None, GT_COLOR,
            )
            cv2.imshow(f"GT scenario {i}  [{trivial_str}]", img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    elif args.visualise:
        if not args.checkpoint:
            parser.error("--visualise requires --checkpoint <path>")

        device = torch.device("cpu")
        model, epoch, cfg = load_checkpoint(args.checkpoint, device)
        model.eval()
        print(f"Loaded checkpoint from epoch {epoch}")

        rng = np.random.default_rng(args.seed)

        for i in range(args.n):
            while True:
                s = generate_scenario(cfg.map.width, cfg.map.height, cfg.train.num_robots, rng)
                if s is not None:
                    break

            # Prepare tensors (batch size 1)
            obs    = torch.tensor(s["obstacle_map"]).unsqueeze(0)                        # (1, H, W)
            masks  = torch.tensor(s["robot_masks"]).unsqueeze(0)                         # (1, N, H, W)
            r_pos  = torch.tensor(s["robot_positions"],  dtype=torch.long).unsqueeze(0)  # (1, N, 2)
            t_pos  = torch.tensor(s["target_positions"], dtype=torch.long).unsqueeze(0)  # (1, N, 2)

            with torch.no_grad():
                pred = model(obs, masks, r_pos, t_pos)[0].cpu().numpy()  # (N, N)

            gt = s["assignment"]
            trivial_str = "trivial" if s["trivial"] else "non-trivial"
            print(f"Scenario {i}  [{trivial_str}]")

            gt_img   = _render_scenario(
                s["obstacle_map"], s["robot_positions"], s["target_positions"],
                gt, GT_COLOR,
            )
            pred_img = _render_scenario(
                s["obstacle_map"], s["robot_positions"], s["target_positions"],
                pred, PRED_COLOR,
            )

            combined = np.hstack([gt_img, pred_img])
            cv2.imshow(f"Scenario {i}  [{trivial_str}]  |  GT (green)  vs  Pred (yellow)", combined)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    elif args.benchmark:
        if not args.checkpoint:
            parser.error("--benchmark requires --checkpoint <path>")
        from scripts.b_navassign.benchmark import run_benchmark
        run_benchmark(
            checkpoint=args.checkpoint,
            n=args.n if args.n != 3 else 10_000,
            batch_sizes=args.batch_sizes,
            seed=args.seed if args.seed is not None else 77,
            use_gpu=not args.cpu,
        )


if __name__ == "__main__":
    main()
