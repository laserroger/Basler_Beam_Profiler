"""Generate fixed spot pictures (PNG + NPY with ground truth) for testing the
CV pipeline.  Run: python tools/make_test_images.py [out_dir]"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from beam_profiler.synthetic import Spot, render, spot_grid

CASES = {
    "single_spot": [Spot(600, 600, sigma_x=15.0, amplitude=0.8)],
    "elliptical_spot": [Spot(600, 600, sigma_x=30.0, sigma_y=12.0, theta=np.radians(30), amplitude=0.8)],
    "grid_4x4": spot_grid(1200, 1200, 4, 4, sigma=10.0),
    "grid_8x8": spot_grid(1200, 1200, 8, 8, sigma=8.0),
}


def main(out_dir: str = "test_images"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    for name, spots in CASES.items():
        img = render(1200, 1200, spots, saturation=65535, rng=rng)
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), img)
        np.save(os.path.join(out_dir, f"{name}.npy"), img)
        truth = [
            dict(x=s.x, y=s.y, sigma_x=s.sigma_x, sigma_y=s.sy, theta=s.theta, amplitude=s.amplitude)
            for s in spots
        ]
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(truth, f, indent=2)
        print(f"{name}: {len(spots)} spot(s) -> {out_dir}/{name}.png/.npy/.json")


if __name__ == "__main__":
    main(*sys.argv[1:2])
