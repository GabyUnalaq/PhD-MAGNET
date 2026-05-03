import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from src.net.map_encoder.config import MapEncoderConfig


def visualise_sample(obs: np.ndarray, src: np.ndarray, cell_size: int = 20) -> np.ndarray:
    """
    Render a training sample: obstacle map with the source cell highlighted.

    obs : (H, W) float32 — obstacle map (1 = obstacle)
    src : (H, W) float32 — source mask  (1 at one cell)
    """
    H, W = obs.shape
    img = np.zeros((H * cell_size, W * cell_size, 3), dtype=np.uint8)

    for r in range(H):
        for c in range(W):
            color = (40, 40, 40) if obs[r, c] == 1 else (210, 210, 210)
            cv2.rectangle(img,
                          (c * cell_size, r * cell_size),
                          ((c + 1) * cell_size - 1, (r + 1) * cell_size - 1),
                          color, -1)

    hits = np.argwhere(src > 0)
    if len(hits):
        sr, sc = hits[0]
        cx, cy = sc * cell_size + cell_size // 2, sr * cell_size + cell_size // 2
        cv2.circle(img, (cx, cy), cell_size // 2, (0, 200, 0), -1)
        cv2.putText(img, "S", (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return img


if __name__ == "__main__":
    import argparse
    from src.net.map_encoder.train import train, evaluate
    from src.net.map_encoder.dataset import generate_sample

    parser = argparse.ArgumentParser(description="MapEncoder — train / evaluate / visualise")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train",     action="store_true", help="Train the encoder on distance field regression")
    mode.add_argument("--test",      action="store_true", help="Evaluate a checkpoint on a held-out set")
    mode.add_argument("--visualise", action="store_true", help="Show training samples (obstacle map + source)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint (--test)")
    parser.add_argument("--n",          type=int, default=2,    help="Number of samples to show (--visualise)")
    parser.add_argument("--seed",       type=int, default=None, help="RNG seed for reproducible samples (--visualise)")
    args = parser.parse_args()

    if args.train:
        train(MapEncoderConfig())

    elif args.test:
        if not args.checkpoint:
            parser.error("--test requires --checkpoint <path>")
        evaluate(args.checkpoint)

    elif args.visualise:
        cfg = MapEncoderConfig()
        rng = np.random.default_rng(args.seed)

        for i in range(args.n):
            seed = int(rng.integers(0, 99999))
            obs, src, gt = generate_sample(
                cfg.train.map_width, cfg.train.map_height, seed=seed
            )
            input_img = visualise_sample(obs, src)

            # Distance field: grayscale, upscaled to match input_img size
            cell_size = 20
            H, W = gt.shape
            gt_gray = (gt * 255).astype(np.uint8)
            gt_img  = cv2.resize(gt_gray, (W * cell_size, H * cell_size), interpolation=cv2.INTER_NEAREST)
            gt_bgr  = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)

            img = np.hstack([input_img, gt_bgr])
            cv2.imshow(f"Sample {i}  (seed {seed})", img)
            print(f"Sample {i}  seed={seed}  free_cells={int((obs == 0).sum())}  "
                  f"source=({np.argwhere(src > 0)[0][0]}, {np.argwhere(src > 0)[0][1]})")
            cv2.waitKey(0)

        cv2.destroyAllWindows()
