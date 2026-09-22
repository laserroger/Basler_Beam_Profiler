"""Compare numerical and exact Gaussian derivatives on identical saved frames.

Run: python -m benchmarks.profiler_cpu data/frame.npy [data/other.npy]
Times the whole fit (including preprocessing), with three warmups and alternating
order to reduce clock/temperature bias. The reference uses the same model,
initialization, bounds and tolerances, but SciPy's numerical derivatives.
"""
import argparse
import json
import time
from unittest.mock import patch

import numpy as np

from beam_profiler.processing import profiler


def benchmark(frame, repeats=30):
    solver = profiler.least_squares

    def numerical(*args, **kwargs):
        kwargs['jac'] = '2-point'
        return solver(*args, **kwargs)

    times = {'numerical': [], 'exact': []}
    results = {}
    for iteration in range(repeats + 3):
        order = ['numerical', 'exact'] if iteration % 2 else ['exact', 'numerical']
        for name in order:
            with patch.object(profiler, 'least_squares', numerical if name == 'numerical' else solver):
                start = time.perf_counter()
                results[name] = profiler.fit_single_beam(frame)
                elapsed = (time.perf_counter() - start) * 1000
            if iteration >= 3:
                times[name].append(elapsed)
    report = {'shape': list(frame.shape), 'repeats': repeats}
    for name, samples in times.items():
        report[name] = {'median_ms': float(np.median(samples)),
                        'p95_ms': float(np.percentile(samples, 95)),
                        'message': results[name].message}
    a, b = results['numerical'].spots, results['exact'].spots
    if len(a) and len(b):
        report['max_center_difference_px'] = float(max(abs(a.x-b.x).max(), abs(a.y-b.y).max()))
        report['max_radius_difference_px'] = float(max(abs(a.sigma_0-b.sigma_0).max(), abs(a.sigma_1-b.sigma_1).max()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', nargs='+')
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    print(json.dumps({p: benchmark(np.load(p, allow_pickle=False), args.repeats)
                      for p in args.frames}, indent=2))


if __name__ == '__main__':
    main()
