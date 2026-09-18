# Why the fitting defaults are what they are

The measurement record behind every default in `beam_profiler/fitconfig.py`.
For *how* the fit works, read [fitting.md](fitting.md) first — this file is
the evidence, not the explanation.

None of the defaults is a guess: each is the minimum of a measured curve, and
`tests/test_fit_config.py` re-runs the measurement, so a default cannot drift
without a test failing.

All figures below are from synthetic frames with known ground truth
(`beam_profiler/synthetic.py`), background 0.01, noise 0.003 of full scale.

## Why the blob radius cannot set the crop

`SimpleBlobDetector` measures its blob on a *globally* Otsu-thresholded mask,
so a faint spot is cut higher up its own profile and comes out smaller. Across
a frame with mixed brightness the ratio `blob_radius / σ` spanned **0.75–1.70**.
Since the crop drives the width estimate, that spread propagated straight into
the result: on 150 scattered spots with amplitudes 0.35–0.9, width error
correlated with `blob_r/σ` at **−0.90** and with amplitude at **−0.81**, and the
worst spot was **−30%**.

Deriving the crop from the fitted σ instead removes the coupling entirely:

| brightness spread | position err (med / max) | width err (med / max) |
|---|---|---|
| uniform (0.8) | 0.012 / 0.029 px | 1.34% / 2.49% |
| mild (0.6–0.9) | 0.013 / 0.033 px | 1.33% / 2.76% |
| wide (0.35–0.9) | 0.015 / 0.055 px | 1.34% / **3.78%** |

(before: 0.046 / 0.474 px and 3.70% / **29.55%**)

## `crop_stages = 3`

Re-cropping converges from any seed, but the worst seed needs three passes:

| seed `r/σ` | stage 0 | stage 1 | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 0.75 | −42.34% | −17.07% | −2.76% | **−0.04%** |
| 1.00 | −24.78% | −3.98% | −1.36% | **−0.04%** |
| 1.55 | −3.98% | −1.12% | −0.04% | −0.04% |
| 1.70 | −1.73% | −0.04% | −0.04% | −0.04% |

A larger `seed_window` looked like it would save a stage, but measured across
σ ∈ {4, 6, 12} and amplitudes {0.2, 0.8} it is both worse and slower:

| | worst seed spread | max abs bias | CPU, 6400 spots |
|---|---:|---:|---:|
| **seed 2.0, stages 3 (default)** | **3.18%** | 2.59% | **383 ms** |
| seed 3.0, stages 2 | 5.77% | 2.59% | 441 ms |
| seed 2.0, stages 2 | 6.58% | 3.06% | — |

A stage whose crop did not move is skipped on both backends — the fit depends
only on the crop, so it would return exactly the same numbers.

## `crop_sigma = 3.5`

Too small truncates the spot; too large lets tail noise dominate. RMS error
over 6 noise realisations, σ = 6:

| N (σ) | bright: bias / spread / **RMS** | faint: bias / spread / **RMS** |
|---:|---|---|
| 3.00 | −4.54 / 0.39 / **4.56** | −4.48 / 1.46 / **4.72** |
| 3.50 | −1.43 / 0.39 / **1.48** | −0.75 / 1.96 / **2.10** ← faint optimum |
| 3.75 | −0.74 / 0.60 / **0.95** | −0.16 / 3.19 / **3.19** |
| 4.00 | +0.15 / 0.62 / **0.64** ← bright optimum | +2.36 / 3.11 / **3.90** |
| 4.50 | −0.00 / 1.13 / **1.13** | −0.13 / 6.19 / **6.19** |

3.5 wins the minimax (2.10% worst case, against 3.90% at 4.0) and the mean.

Half-widths are taken **per axis** from `√cxx` and `√cyy`, not from one radius
off the major axis. On an 18×5 elliptical spot:

| | crop | σ₀ err | σ₁ err |
|---|---|---:|---:|
| major axis for both | 144×144 = 20 736 px | +0.38% | +6.25% |
| per-axis (default) | 143×40 = **5 720 px** | −0.28% | **+0.01%** |

## `clip_negative = False`

This one matters more than it looks. After background subtraction a tail pixel
is pure noise with `E[z] = 0`, so it contributes nothing on average. Clipping
makes `E[max(z,0)] = σ_noise/√(2π) = 0.399·σ_noise` — measured 0.3993 over 2M
samples — so **every** tail pixel gains a positive offset, which the second
moment then multiplies by `d²`. The spurious term grows like `L⁴`:

| crop half-width L | clip reports | true 2σ | the window's own 2σ (`2L/√3`) | ratio |
|---:|---:|---:|---:|---:|
| 24 | 14.17 | 12.00 | 27.71 | 0.511 |
| 96 | 86.35 | 12.00 | 110.85 | 0.779 |
| 384 | 433.79 | 12.00 | 443.41 | **0.978** |

Widen the window far enough with clipping on and you measure the *crop*, not
the beam. The setting exists only to reproduce pre-2026 numbers.

## `adaptive_iters` — 0 by default

`0` gives plain second moments (the ISO 11146 D4σ definition). `> 0` iterates a
Gaussian weight matched to the current covariance: for a Gaussian source seen
through a Gaussian weight the measured covariance is `(C⁻¹ + W⁻¹)⁻¹`, so at the
fixed point `W = C` it is exactly `C/2`, hence **`C = 2M`** — an analytic
relation, not a calibration.

Weighting is safe where clipping is not because `w·z` is *linear* in `z`:
`E[w·z] = w·E[z] = 0`, so the noise still cancels. Clipping changes the
noise's expectation; weighting changes only its variance.

σ error (mean ± spread over 24 realisations), bright:

| | 3σ | 4σ | 5σ | 6σ | 8σ | 12σ |
|---|---|---|---|---|---|---|
| clip | −3.69 ±0.22 | +2.09 ±0.53 | +7.66 ±1.01 | +17.31 ±1.66 | +48.65 ±2.89 | +160.27 ±5.24 |
| no clip | −4.54 ±0.30 | −0.12 ±0.86 | +0.60 ±2.02 | +2.53 ±3.72 | +4.52 ±8.35 | +0.78 ±39.30 |
| weighted | −0.98 ±0.10 | +0.02 ±0.10 | +0.05 ±0.08 | +0.08 ±0.08 | +0.06 ±0.05 | — |

Clip: small spread, unbounded bias — systematic, so averaging frames will not
reveal it. No clip: unbiased but the variance grows with the window. Weighted:
both controlled, and the crop choice stops mattering over 3.5σ–8σ.

It converges linearly at ≈0.5 per iteration, so it is not free:
1 iter +4.7%, 2 → +2.3%, 4 → +0.60%, 6 → +0.20%, 8 → +0.10%.

## Cost (4070 SUPER, 4096², 6400 spots)

| | detect + fit | whole frame |
|---|---:|---:|
| moments, CUDA | **34.5 ms** | 70 ms (14.3 fps) |
| moments, CPU | 377 ms | 410 ms (2.4 fps) |
| adaptive ×8, CUDA | 94 ms | 125 ms (8.0 fps) |
| adaptive ×8, CPU | 5 001 ms | 5 032 ms (0.2 fps) |

Adaptive is viable on the GPU and not on the CPU. On a CPU-only machine,
lowering `crop_stages` is the knob to reach for.

## Verification

`fit.fit_spot` is a scalar reference implementation. `fit.fit_spots` (numpy,
batched) and the fused CUDA kernel in `gpu.py` must both reproduce it exactly:
`tests/test_processing.py` asserts agreement to **1e-9** across five
configurations, and the same check has been run over uint8/uint16 frames,
non-square sensors, frame-edge spots, elliptical spots and single-pixel spots,
with worst observed deviations of 0 px in position and ~7e-13 in width.
Any change to one path must be mirrored in the other two.
