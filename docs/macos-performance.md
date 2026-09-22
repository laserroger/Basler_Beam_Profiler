# macOS single-beam performance

Measured locally on 2026-09-21: Apple M5 Pro, native arm64 Python 3.12,
NumPy linked to Apple Accelerate. These timings cover `fit_single_beam`, not
camera acquisition, rendering, or the viewer's total FPS.

The profiler now supplies exact Gaussian derivatives to SciPy. Residual and
Jacobian evaluations share intermediate results within each fit; the Jacobian
uses column-contiguous storage for its linear algebra. The Gaussian model,
128-pixel sampling limit, parameter bounds, solver tolerances and float64 CPU
calculations are unchanged. There is no reuse of a previous frame's fit.

## Measurements

Thirty measured fits after three warmups, alternating CPU implementations:

| Saved frame | Numerical derivatives | Exact derivatives | Fit time reduction |
| --- | ---: | ---: | ---: |
| 2048 × 1536 broad beam | 19.12 ms | 13.26 ms | 31% |
| 588 × 436 core and halo | 21.04 ms | 13.94 ms | 34% |

Maximum differences between the two CPU implementations were below 0.000001
pixel for both center coordinates and radii. These figures compare the same
least-squares objective; they do not measure the separate correction from
robust-loss fitting to ordinary intensity least squares.

Reproduce using local raw captures (not committed to the repository):

```sh
python -m benchmarks.profiler_cpu data/frame.npy data/other-frame.npy
```

## Apple GPU experiment

A compiled MLX 0.32.2 evaluator was run on the actual Apple M5 Pro GPU.
It calculates Gaussian residuals and exact derivatives on Metal, returning
them to the same SciPy optimizer. Measured median times were 20.69 ms for the
broad beam and 23.45 ms for the core-and-halo frame, including synchronization
and array conversion. This path was slower than the optimized CPU and is
therefore retained only as an optional benchmark, not enabled in the viewer.
MLX is not an application dependency.

```sh
# In a separate benchmark environment with MLX available:
python -m benchmarks.profiler_metal data/frame.npy data/other-frame.npy
```

The experiment uses float32 GPU calculations and float64 SciPy calculations.
It does not establish that all Metal implementations are slower: an optimizer
entirely on the GPU or a large batch of beams is a different workload. Array
spot fitting still uses the existing CPU/CUDA paths. Apple Accelerate is a
CPU math library, not an Apple GPU backend.

References: [SciPy callable Jacobians](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html),
[MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html),
[MLX unified memory and device selection](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).
