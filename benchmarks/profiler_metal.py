"""Experimental Metal evaluator benchmark; not imported by the application.

Install MLX separately, then run from the repository root:
    python -m benchmarks.profiler_metal data/frame.npy
Includes NumPy/Metal synchronization and the full SciPy solve. GPU timing is
not a measurement of a fully GPU-resident optimizer. Uses float32 on Metal.
"""
import argparse
import time

import mlx.core as mx
import numpy as np

from beam_profiler.processing import profiler


@mx.compile
def evaluate(x, y, z, p):
    bg, amplitude, cx, cy, lx, ly, angle = (p[i] for i in range(7))
    c, s = mx.cos(angle), mx.sin(angle)
    u, v = c*(x-cx)+s*(y-cy), -s*(x-cx)+c*(y-cy)
    ix, iy = mx.exp(-2*lx), mx.exp(-2*ly)
    e = mx.exp(-.5*(u*u*ix+v*v*iy))
    signal = amplitude*e
    residual = bg+signal-z
    jac = mx.stack([mx.ones_like(x), e, signal*(c*u*ix-s*v*iy),
                    signal*(s*u*ix+c*v*iy), signal*u*u*ix,
                    signal*v*v*iy, signal*u*v*(iy-ix)], axis=1)
    return residual, jac


class MetalEvaluator:
    def __init__(self, x, y, z):
        self.xyz = tuple(mx.array(a.astype(np.float32)) for a in (x, y, z))
        self.p = None

    def _evaluate(self, p):
        if self.p is not None and np.array_equal(self.p, p):
            return
        r, j = evaluate(*self.xyz, mx.array(p.astype(np.float32)))
        mx.eval(r, j)
        self.r, self.j = np.array(r, dtype=np.float64), np.array(j, dtype=np.float64)
        self.p = p.copy()

    def residual(self, p):
        self._evaluate(p)
        return self.r

    def jacobian(self, p):
        self._evaluate(p)
        return self.j


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', nargs='+')
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    print('MLX', mx.__version__, mx.device_info())
    mx.set_default_device(mx.gpu)
    cpu = profiler._GaussianEvaluator
    try:
        for path in args.frames:
            frame = np.load(path, allow_pickle=False)
            for name, evaluator in [('CPU exact', cpu), ('Metal compiled', MetalEvaluator)]:
                profiler._GaussianEvaluator = evaluator
                for _ in range(3):
                    profiler.fit_single_beam(frame)
                elapsed = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    result = profiler.fit_single_beam(frame)
                    elapsed.append(1000*(time.perf_counter()-start))
                print(path, name, 'median_ms', np.median(elapsed), 'p95_ms', np.percentile(elapsed, 95),
                      'center', result.spots.x, result.spots.y, 'radii', result.spots.sigma_0, result.spots.sigma_1,
                      'status', result.message, flush=True)
    finally:
        profiler._GaussianEvaluator = cpu


if __name__ == '__main__':
    main()
