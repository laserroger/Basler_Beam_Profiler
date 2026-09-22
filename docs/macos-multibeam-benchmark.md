# Synthetic multi-beam benchmark on Apple M5 Pro

Measured 2026-09-21 using MLX 0.32.2 and the application's batched NumPy CPU fitter.
The Metal implementation is an isolated benchmark prototype, not enabled in the app.

## Workload and timing

- Fixed 2048 × 1536, unsigned 16-bit frames.
- 1, 4, 16, 64, 256, or 1024 rotated elliptical Gaussian beams; randomized amplitudes,
  widths and subpixel centers; background 500 DN and Gaussian noise of 12 DN RMS.
- Candidate centers are deliberately offset by up to half a pixel; candidate radii vary.
- Identical moment estimator: border-median background subtraction, three re-crops,
  principal axes and final intensity statistics. Adaptive weighting tested at 0 and 4 iterations.
- Crop size capped at 48 pixels for this small-spot workload. This is not a test of
  arbitrary large crops, overlapping beams, saturation, or a nonlinear Gaussian fit per beam.
- CPU uses production float64 accumulation. Metal uses batched float32 operations,
  variable-size crops padded to the cap, and local coordinates to limit cancellation.
- CPU retains its existing optimization to skip unchanged intermediate crops; the
  prototype recomputes every stage, producing equivalent results on these frames.
- Ten warmups, then 40 repetitions per case, alternating CPU/GPU order. Medians below.
- Timings include neighbor search, frame input conversion/upload, the fit, GPU
  synchronization, and output conversion. Synthetic generation, blob detection,
  camera acquisition, rendering and first-use initialization are excluded.
- The existing viewer remained running, so background load and clock behavior can
  affect small differences. Treat near-equal timings as a tie, not a universal threshold.

## Results

| Beams | CPU, plain (ms) | Metal, plain (ms) | CPU, 4 adaptive (ms) | Metal, 4 adaptive (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.25 | 1.89 | 0.56 | 7.20 |
| 4 | 0.47 | 1.79 | 1.51 | 6.02 |
| 16 | 0.99 | 2.83 | 3.79 | 6.94 |
| 64 | 2.00 | 2.02 | 9.22 | 7.63 |
| 256 | 4.63 | 3.62 | 23.32 | 7.16 |
| 1024 | 14.87 | 5.73 | 74.78 | 22.86 |

At 1024 beams, Metal was 2.60× faster for plain moments and 3.27× faster for
four adaptive iterations. Plain moments were essentially tied at 64 beams and
Metal was modestly faster at 256. Adaptive weighting benefited at smaller batch
sizes. These are fitting speedups, not whole-viewer FPS predictions.

## Accuracy and reproducibility

Every case checked the same candidate count, valid fits, center, radii, principal
axis orientation and both intensity outputs against the CPU. Largest CPU/Metal
center difference: 0.0000637 pixel; largest radius difference: 0.00000148 pixel.
Against the known synthetic truth, CPU radius RMS relative error was about 1%
for plain moments and 0.1–0.15% for adaptive weighting. This shared estimator
bias is much larger than the CPU/Metal numerical difference.

Raw timing, 95th percentiles, configuration and accuracy results:
[m5-pro-multibeam.json](../benchmarks/results/m5-pro-multibeam.json).

```sh
# MLX is optional and only needed in the benchmark environment.
python -m benchmarks.multibeam_metal --repeats 40 --output results.json
```

The local generated example and ground truth are in
`data/validation/synthetic_1024_beams.npy` and
`data/validation/synthetic_1024_beams_truth.npz`; the adjacent PNG is a preview.
The deterministic generator in the script reproduces all workloads with seed 412.
