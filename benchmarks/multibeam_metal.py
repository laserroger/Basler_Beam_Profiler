"""Synthetic full-frame CPU vs batched Metal moment fitting benchmark.

Optional MLX dependency; never imported by the application. Run from repo root:
  python -m benchmarks.multibeam_metal --output /tmp/multibeam-results.json

Both paths fit the same seeded candidates with background subtraction, three
re-crops, and final intensity/ellipse statistics. CPU is the production batched
NumPy fitter. Metal is a batched float32 prototype using padded variable-size
crops. Timing includes neighbour search, input conversion/upload, fitting, GPU
synchronization and output conversion; excludes candidate detection and UI.
"""
import argparse
from dataclasses import asdict
import json
import time

import mlx.core as mx
import numpy as np

from beam_profiler.fitconfig import FitConfig
from beam_profiler.processing.fit import fit_spots, neighbour_limit


def synthetic(n, seed=412):
    rng = np.random.default_rng(seed)
    h, w, pitch = 1536, 2048, 48
    frame = rng.normal(500, 12, (h, w)).astype(np.float32)
    slots = np.array([(x, y) for y in range(32, h-32, pitch)
                      for x in range(32, w-32, pitch)])
    slots = slots[rng.permutation(len(slots))[:n]]
    truth = []
    for x, y in slots:
        cx, cy = x+rng.uniform(-.4, .4), y+rng.uniform(-.4, .4)
        sx, sy, angle = rng.uniform(3.5, 4.8), rng.uniform(2.5, 3.4), rng.uniform(-3, 3)
        yy, xx = np.mgrid[y-22:y+23, x-22:x+23]
        u = np.cos(angle)*(xx-cx)+np.sin(angle)*(yy-cy)
        v = -np.sin(angle)*(xx-cx)+np.cos(angle)*(yy-cy)
        frame[y-22:y+23, x-22:x+23] += rng.uniform(15000, 45000)*np.exp(-.5*((u/sx)**2+(v/sy)**2))
        truth.append((cx, cy, 2*sx, 2*sy))
    truth = np.array(truth)
    seeds = truth[:, :2] + rng.uniform(-.5, .5, (n, 2))
    radii = rng.uniform(4, 7, n)
    return np.clip(np.rint(frame), 0, 65535).astype(np.uint16), seeds[:, 0], seeds[:, 1], radii, truth


def make_metal_fit(cfg):
    side = cfg.max_crop
    q = cfg.crop_quantum

    def fit(frame, cx, cy, radius, cap):
        h, w = frame.shape
        ar = mx.arange(side)
        ar_border = mx.arange(4*side)
        hx = mx.minimum(cfg.seed_window*radius, cap)
        hy = hx
        alive = mx.ones_like(cx, dtype=mx.bool_)
        for stage in range(cfg.crop_stages+1):
            cw = mx.clip(mx.ceil(2*hx/q)*q, cfg.min_crop, min(side, w)).astype(mx.int32)
            ch = mx.clip(mx.ceil(2*hy/q)*q, cfg.min_crop, min(side, h)).astype(mx.int32)
            x0 = mx.clip(mx.round(cx).astype(mx.int32)-cw//2, 0, w-cw)
            y0 = mx.clip(mx.round(cy).astype(mx.int32)-ch//2, 0, h-ch)
            xs = mx.minimum(ar[None, :], cw[:, None]-1)
            ys = mx.minimum(ar[None, :], ch[:, None]-1)
            z = frame[(y0[:, None]+ys)[:, :, None], (x0[:, None]+xs)[:, None, :]].astype(mx.float32)
            valid = (ar[None, :, None] < ch[:, None, None]) & (ar[None, None, :] < cw[:, None, None])
            if cfg.subtract_background:
                # Exactly the CPU border convention, including doubled corners.
                j = ar_border[None, :]
                bx = mx.where(j < cw[:, None], j,
                     mx.where(j < 2*cw[:, None], j-cw[:, None],
                     mx.where(j < (2*cw+ch)[:, None], 0, cw[:, None]-1)))
                by = mx.where(j < cw[:, None], 0,
                     mx.where(j < 2*cw[:, None], ch[:, None]-1,
                     mx.where(j < (2*cw+ch)[:, None], j-2*cw[:, None], j-(2*cw+ch)[:, None])))
                length = 2*(cw+ch)
                border = frame[mx.clip(y0[:, None]+by, 0, h-1), mx.clip(x0[:, None]+bx, 0, w-1)].astype(mx.float32)
                border = mx.sort(mx.where(j < length[:, None], border, float('inf')), axis=1)
                lo = mx.take_along_axis(border, ((length-1)//2)[:, None], axis=1)[:, 0]
                hi = mx.take_along_axis(border, (length//2)[:, None], axis=1)[:, 0]
                z = z - (.5*(lo+hi))[:, None, None]
            if cfg.clip_negative:
                z = mx.maximum(z, 0)
            z = mx.where(valid, z, 0)
            col, row = mx.sum(z, axis=1), mx.sum(z, axis=2)
            total = mx.sum(col, axis=1)
            alive = alive & (total > 0)
            total = mx.where(total > 0, total, 1)
            lx = mx.sum(col*ar, axis=1)/total
            ly = mx.sum(row*ar, axis=1)/total
            dx, dy = ar[None, :]-lx[:, None], ar[None, :]-ly[:, None]
            cxx = mx.sum(col*dx*dx, axis=1)/total
            cyy = mx.sum(row*dy*dy, axis=1)/total
            cxy = mx.sum(z*dy[:, :, None]*dx[:, None, :], axis=(1, 2))/total
            for _ in range(cfg.adaptive_iters):
                det = cxx*cyy-cxy*cxy
                ok = alive & (det > 0)
                safe = mx.where(ok, det, 1)
                quad = (cyy[:, None, None]*dx[:, None, :]**2+cxx[:, None, None]*dy[:, :, None]**2
                        -2*cxy[:, None, None]*dx[:, None, :]*dy[:, :, None])/safe[:, None, None]
                wz = mx.exp(-.5*mx.clip(quad, 0, 200))*z
                total = mx.sum(wz, axis=(1, 2))
                ok = ok & (total > 0)
                total = mx.where(ok, total, 1)
                nx = mx.sum(wz*ar[None, None, :], axis=(1, 2))/total
                ny = mx.sum(wz*ar[None, :, None], axis=(1, 2))/total
                ddx, ddy = ar[None, :]-nx[:, None], ar[None, :]-ny[:, None]
                axx = 2*mx.sum(wz*ddx[:, None, :]**2, axis=(1, 2))/total
                ayy = 2*mx.sum(wz*ddy[:, :, None]**2, axis=(1, 2))/total
                axy = 2*mx.sum(wz*ddy[:, :, None]*ddx[:, None, :], axis=(1, 2))/total
                lx, ly = mx.where(ok, nx, lx), mx.where(ok, ny, ly)
                cxx, cyy, cxy = mx.where(ok, axx, cxx), mx.where(ok, ayy, cyy), mx.where(ok, axy, cxy)
                dx, dy = ar[None, :]-lx[:, None], ar[None, :]-ly[:, None]
            cx, cy = x0+lx, y0+ly
            if stage != cfg.crop_stages:
                hx = mx.minimum(cfg.crop_sigma*mx.sqrt(mx.maximum(cxx, 1e-12)), cap)
                hy = mx.minimum(cfg.crop_sigma*mx.sqrt(mx.maximum(cyy, 1e-12)), cap)
        half = .5*(cxx+cyy)
        disc = mx.sqrt(mx.maximum(.25*(cxx-cyy)**2+cxy*cxy, 0))
        major, minor = mx.sqrt(mx.maximum(4*(half+disc), 0)), mx.sqrt(mx.maximum(4*(half-disc), 0))
        vx, vy = cxy, half+disc-cxx
        norm = mx.sqrt(vx*vx+vy*vy)
        safe = mx.where(norm < 1e-12, 1, norm)
        vx, vy = mx.where(norm < 1e-12, 1, vx/safe), mx.where(norm < 1e-12, 0, vy/safe)
        ix, iy = mx.argmin(mx.abs(dx), axis=1), mx.argmin(mx.abs(dy), axis=1)
        i0 = z[mx.arange(cx.shape[0]), iy, ix]
        det = cxx*cyy-cxy*cxy
        safe = mx.where(det > 0, det, 1)
        quad = (cyy[:, None, None]*dx[:, None, :]**2+cxx[:, None, None]*dy[:, :, None]**2
                -2*cxy[:, None, None]*dy[:, :, None]*dx[:, None, :])/safe[:, None, None]
        inside = (quad <= 1) & valid
        count = mx.sum(inside, axis=(1, 2))
        iw = mx.where((det > 0) & (count > 0), mx.sum(mx.where(inside, z, 0), axis=(1, 2))/mx.maximum(count, 1)/(2*(1-np.exp(-.5))), i0)
        return mx.stack([cx, cy, major, minor, vx, vy, i0, iw, alive.astype(mx.float32)], axis=1)

    def run(frame, x, y, r):
        cap = neighbour_limit(x, y, cfg)
        output = fit(mx.array(frame), *(mx.array(a.astype(np.float32)) for a in (x, y, r, cap)))
        mx.eval(output)
        return np.array(output)
    return run


def cpu_fit(frame, x, y, r, cfg):
    s = fit_spots(frame, x, y, r, cfg)
    return np.column_stack([s.x, s.y, s.sigma_0, s.sigma_1, s.vec_0, s.I0, s.I0_weighted, np.ones(len(s))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', nargs='+', type=int, default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument('--iterations', nargs='+', type=int, default=[0, 4])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 or n > 1302 for n in args.counts):
        parser.error('Use positive repeats and beam counts between 1 and 1302')
    if any(i < 0 for i in args.iterations):
        parser.error('Adaptive iterations must be nonnegative')
    mx.set_default_device(mx.gpu)
    report = {'device': mx.device_info(), 'mlx': mx.__version__, 'rows': [], 'repeats': args.repeats, 'warmups': 10,
              'frame_shape': [1536, 2048], 'seed': 412,
              'scope': 'Seeded multi-spot moment fit including upload and synchronization; detection and UI excluded'}
    for iters in args.iterations:
        cfg = FitConfig(gpu_enabled=False, max_crop=48, adaptive_iters=iters)
        metal = make_metal_fit(cfg)
        for n in args.counts:
            f, x, y, r, truth = synthetic(n)
            funcs = {'cpu': lambda: cpu_fit(f, x, y, r, cfg), 'metal': lambda: metal(f, x, y, r)}
            timing = {key: [] for key in funcs}
            results = {}
            for repeat in range(args.repeats+10):
                for key in (['cpu', 'metal'] if repeat%2 else ['metal', 'cpu']):
                    start = time.perf_counter()
                    results[key] = funcs[key]()
                    elapsed = 1000*(time.perf_counter()-start)
                    if repeat >= 10:
                        timing[key].append(elapsed)
            a, b = results['cpu'], results['metal']
            assert a.shape == b.shape == (n, 9)
            assert np.all(b[:, -1] == 1)
            center_error = float(np.max(np.abs(a[:, :2]-b[:, :2])))
            radius_error = float(np.max(np.abs(a[:, 2:4]-b[:, 2:4])))
            # Tolerances are explicitly looser than CUDA's float64 parity.
            assert center_error < .002, center_error
            assert radius_error < .005, radius_error
            intensity_error = float(np.max(np.abs(a[:, 6:8]-b[:, 6:8])/np.maximum(np.abs(a[:, 6:8]), 1)))
            axis_error = float(np.max(np.maximum(0, 1-np.abs(np.sum(a[:, 4:6]*b[:, 4:6], axis=1)))))
            assert intensity_error < .005, intensity_error
            assert axis_error < .001, axis_error
            row = {'beams': n, 'adaptive_iterations': iters,
                   'cpu_ms': float(np.median(timing['cpu'])), 'metal_ms': float(np.median(timing['metal'])),
                   'cpu_p95_ms': float(np.percentile(timing['cpu'], 95)), 'metal_p95_ms': float(np.percentile(timing['metal'], 95)),
                   'max_center_difference_px': center_error, 'max_radius_difference_px': radius_error,
                   'max_intensity_relative_difference': intensity_error, 'max_axis_error': axis_error,
                   'cpu_center_rms_vs_truth_px': float(np.sqrt(np.mean((a[:, :2]-truth[:, :2])**2))),
                   'cpu_radius_relative_rms_vs_truth': float(np.sqrt(np.mean(((a[:, 2:4]-truth[:, 2:4])/truth[:, 2:4])**2))),
                   'config': asdict(cfg)}
            row['speedup'] = row['cpu_ms']/row['metal_ms']
            report['rows'].append(row)
            print(json.dumps(row), flush=True)
            with open(args.output, 'w') as out:
                json.dump(report, out, indent=2)


if __name__ == '__main__':
    main()
