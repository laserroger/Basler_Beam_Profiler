"""Entry point: python -m beam_profiler [--sim] [options]."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="beam_profiler", description="Basler Beam Profiler"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="use the simulated camera (fixed spot grid) instead of hardware",
    )
    parser.add_argument(
        "--mode", choices=["8Bit", "16Bit"], default="16Bit", help="pixel depth"
    )
    parser.add_argument(
        "--sim-grid", metavar="NxM", default="4x4",
        help="simulated spot grid, e.g. 4x4 (default) or 8x8",
    )
    parser.add_argument(
        "--sim-size", metavar="WxH", default="1200x1200", help="simulated sensor size"
    )
    parser.add_argument(
        "--sim-jitter", type=float, default=0.3,
        help="per-frame spot position jitter in px (0 = perfectly static)",
    )
    parser.add_argument(
        "--sim-sigma", type=float, default=None,
        help="spot sigma in px (default: pitch/10, i.e. waist = pitch/5)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    from .cameras import SimulatedCamera, open_cameras

    grid = tuple(int(v) for v in args.sim_grid.lower().split("x"))
    width, height = (int(v) for v in args.sim_size.lower().split("x"))
    sim_opts = dict(
        grid=grid, width=width, height=height,
        jitter=args.sim_jitter, sigma=args.sim_sigma,
    )
    cameras = open_cameras(mode=args.mode, simulate=args.sim, **sim_opts)

    from .ui import Viewer

    Viewer(cameras).run()


if __name__ == "__main__":
    main()
