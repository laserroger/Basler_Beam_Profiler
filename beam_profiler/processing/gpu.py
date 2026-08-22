"""Optional CuPy backend for the batched spot fit.

Everything here is best-effort: if CuPy, a driver or a usable device is
missing, `fit_batch` returns None and the caller falls back to numpy.  Import
of this module never imports CuPy - the probe happens on first use so the
CUDA context cost (~0.8 s) is only paid by machines that have one.

Why a hand-written kernel instead of stacked CuPy array ops: the per-spot work
is tiny (a ~35x35 crop), so the GPU spends its time waiting for Python to issue
the next launch.  The array-op version needs ~50 kernel launches per distinct
crop size; the fused kernel below is a single launch for all spots, handles a
different crop size per spot, and keeps every intermediate in registers.

Set BEAM_PROFILER_GPU=0 to force the CPU path, or =force to use the GPU even
below the spot-count threshold (useful for benchmarking).
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

from .spots import SpotArray

# below this many spots the fixed cost (frame upload + launch) is not worth it;
# measured crossover on a 4070 SUPER is ~300 spots
MIN_SPOTS = 400
MAX_CROP = 96  # kernel shared-memory limit on the crop side
BLOCK = 128  # threads per block, must be a power of two

_cupy = None  # None = not probed yet, False = unusable
_kernels: dict = {}
_frame_buf = None
_probe_lock = threading.Lock()
_last_backend = "cpu"  # which path actually ran the last batch, for the HUD

_KERNEL_SOURCE = r"""
#define BLOCK %(block)d
#define MAX_BORDER %(max_border)d

__device__ __forceinline__ double block_sum(double v, double *sh)
{
    int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    double r = sh[0];
    __syncthreads();
    return r;
}

// index of the sample nearest to c, ties going to the lower index (numpy argmin)
__device__ __forceinline__ int nearest_index(double c, int n)
{
    int lo = (int)floor(c);
    if (lo < 0) return 0;
    if (lo >= n - 1) return n - 1;
    return (fabs(c - (double)lo) <= fabs(c - (double)(lo + 1))) ? lo : lo + 1;
}

#define FIT_KERNEL(NAME, PIXEL)                                                     \
extern "C" __global__ void NAME(                                                    \
    const PIXEL *__restrict__ img, const int W,                                     \
    const long long *__restrict__ x0a, const long long *__restrict__ y0a,           \
    const int *__restrict__ cwa, const int *__restrict__ cha,                       \
    double *__restrict__ out, const int n)                                          \
{                                                                                   \
    const int b = blockIdx.x;                                                       \
    if (b >= n) return;                                                             \
    const int t = threadIdx.x;                                                      \
    const int x0 = (int)x0a[b], y0 = (int)y0a[b];                                    \
    const int cw = cwa[b], ch = cha[b];                                             \
    const int npix = cw * ch;                                                        \
                                                                                    \
    __shared__ double sh[BLOCK];                                                    \
    __shared__ float border[MAX_BORDER];                                            \
    __shared__ float med_lo, med_hi;                                                \
                                                                                    \
    /* crop border, corners counted twice (matches the numpy concatenate) */        \
    const int m = 2 * (cw + ch);                                                    \
    for (int i = t; i < m; i += BLOCK) {                                            \
        float v;                                                                    \
        if (i < cw)               v = img[(long long)y0 * W + x0 + i];              \
        else if (i < 2 * cw)      v = img[(long long)(y0 + ch - 1) * W + x0 + i - cw]; \
        else if (i < 2 * cw + ch) v = img[(long long)(y0 + i - 2 * cw) * W + x0];   \
        else                      v = img[(long long)(y0 + i - 2 * cw - ch) * W + x0 + cw - 1]; \
        border[i] = v;                                                              \
    }                                                                               \
    if (t == 0) { med_lo = 0.0f; med_hi = 0.0f; }                                   \
    __syncthreads();                                                                \
                                                                                    \
    /* median by rank counting: v is the k-th smallest iff lt <= k < lt + eq */     \
    const int k_lo = (m - 1) / 2, k_hi = m / 2;                                     \
    for (int i = t; i < m; i += BLOCK) {                                            \
        const float v = border[i];                                                  \
        int lt = 0, eq = 0;                                                         \
        for (int j = 0; j < m; ++j) { float u = border[j]; lt += (u < v); eq += (u == v); } \
        if (lt <= k_lo && k_lo < lt + eq) med_lo = v;                                \
        if (lt <= k_hi && k_hi < lt + eq) med_hi = v;                                \
    }                                                                               \
    __syncthreads();                                                                \
    const float bg = (m & 1) ? med_lo : (med_lo + med_hi) * 0.5f;                    \
                                                                                    \
    /* pass 1: total intensity and first moments */                                 \
    double S = 0.0, Sx = 0.0, Sy = 0.0;                                             \
    for (int p = t; p < npix; p += BLOCK) {                                         \
        const int r = p / cw, c = p - r * cw;                                       \
        float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;                \
        if (!(z > 0.0f)) continue;                                                  \
        const double zd = (double)z;                                                \
        S += zd; Sx += zd * (double)(x0 + c); Sy += zd * (double)(y0 + r);          \
    }                                                                               \
    S = block_sum(S, sh); Sx = block_sum(Sx, sh); Sy = block_sum(Sy, sh);           \
    if (!(S > 0.0)) { if (t == 0) out[b * 9 + 8] = 0.0; return; }                   \
    const double mx = Sx / S, my = Sy / S;                                          \
                                                                                    \
    /* pass 2: central second moments */                                            \
    double cxx = 0.0, cyy = 0.0, cxy = 0.0;                                         \
    for (int p = t; p < npix; p += BLOCK) {                                         \
        const int r = p / cw, c = p - r * cw;                                       \
        float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;                \
        if (!(z > 0.0f)) continue;                                                  \
        const double zd = (double)z;                                                \
        const double dx = (double)(x0 + c) - mx, dy = (double)(y0 + r) - my;        \
        cxx += zd * dx * dx; cyy += zd * dy * dy; cxy += zd * dx * dy;              \
    }                                                                               \
    cxx = block_sum(cxx, sh) / S;                                                   \
    cyy = block_sum(cyy, sh) / S;                                                   \
    cxy = block_sum(cxy, sh) / S;                                                   \
                                                                                    \
    /* pass 3: mean intensity inside the 1-sigma ellipse */                         \
    const double det = cxx * cyy - cxy * cxy;                                       \
    double e_sum = 0.0, e_cnt = 0.0;                                                \
    if (det > 0.0) {                                                                \
        const double ia = cyy / det, ib = cxy / det, ic = cxx / det;                \
        for (int p = t; p < npix; p += BLOCK) {                                     \
            const int r = p / cw, c = p - r * cw;                                   \
            float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;            \
            if (!(z > 0.0f)) z = 0.0f;                                              \
            const double dx = (double)(x0 + c) - mx, dy = (double)(y0 + r) - my;    \
            if (ia * dx * dx + ic * dy * dy - 2.0 * ib * dy * dx <= 1.0) {          \
                e_sum += (double)z; e_cnt += 1.0;                                   \
            }                                                                       \
        }                                                                           \
        e_sum = block_sum(e_sum, sh);                                               \
        e_cnt = block_sum(e_cnt, sh);                                               \
    }                                                                               \
                                                                                    \
    if (t != 0) return;                                                             \
    const int col = nearest_index(mx - (double)x0, cw);                             \
    const int row = nearest_index(my - (double)y0, ch);                             \
    float zc = (float)img[(long long)(y0 + row) * W + x0 + col] - bg;               \
    if (!(zc > 0.0f)) zc = 0.0f;                                                    \
    const double I0 = (double)zc;                                                   \
    const double I0w = (det > 0.0 && e_cnt > 0.0)                                   \
        ? e_sum / (e_cnt * %(i0w_norm).17g) : I0;                                   \
                                                                                    \
    const double half_tr = 0.5 * (cxx + cyy);                                       \
    double disc = 0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy;                     \
    disc = disc > 0.0 ? sqrt(disc) : 0.0;                                           \
    const double l_max = half_tr + disc, l_min = half_tr - disc;                    \
    double vx = cxy, vy = l_max - cxx;                                              \
    const double vn = sqrt(vx * vx + vy * vy);                                      \
    if (vn < 1e-12) { vx = 1.0; vy = 0.0; } else { vx /= vn; vy /= vn; }            \
                                                                                    \
    out[b * 9 + 0] = mx;                                                            \
    out[b * 9 + 1] = my;                                                            \
    out[b * 9 + 2] = sqrt(l_max > 0.0 ? 4.0 * l_max : 0.0);                         \
    out[b * 9 + 3] = sqrt(l_min > 0.0 ? 4.0 * l_min : 0.0);                         \
    out[b * 9 + 4] = vx;                                                            \
    out[b * 9 + 5] = vy;                                                            \
    out[b * 9 + 6] = I0;                                                            \
    out[b * 9 + 7] = I0w;                                                           \
    out[b * 9 + 8] = 1.0;                                                           \
}

FIT_KERNEL(fit_spots_u8, unsigned char)
FIT_KERNEL(fit_spots_u16, unsigned short)
FIT_KERNEL(fit_spots_f32, float)
"""

_ENTRY = {np.dtype(np.uint8): "fit_spots_u8",
          np.dtype(np.uint16): "fit_spots_u16",
          np.dtype(np.float32): "fit_spots_f32"}


def _env(name: str) -> str:
    return os.environ.get(name, "").strip().lower()


def _probe():
    """Import CuPy and compile the kernel once; store False if anything fails."""
    global _cupy
    if _cupy is not None:
        return _cupy
    with _probe_lock:
        if _cupy is None:
            _cupy = _do_probe()
    return _cupy


def _do_probe():
    if _env("BEAM_PROFILER_GPU") in ("0", "off", "no", "false"):
        return False
    try:
        import cupy  # noqa: PLC0415  (deliberately deferred, see module docstring)

        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device")
        source = _KERNEL_SOURCE % {
            "block": BLOCK,
            "max_border": 4 * MAX_CROP,
            "i0w_norm": 2 * (1 - np.exp(-0.5)),
        }
        # no --use_fast_math: the fit must stay bit-comparable to the CPU path
        module = cupy.RawModule(code=source)
        for entry in _ENTRY.values():
            _kernels[entry] = module.get_function(entry)
        cupy.zeros(1).sum()  # force context creation now, not mid-frame
        logging.info("GPU spot fitting enabled on device %d", cupy.cuda.Device().id)
        return cupy
    except Exception as e:  # missing cupy, no driver, nvrtc failure, ...
        logging.info("GPU spot fitting unavailable (%s); using the CPU path", e)
        return False


def warm_up():
    """Probe in the background so the first dense frame does not stall for the
    ~0.8 s CuPy import + CUDA context creation."""
    if _cupy is None:
        threading.Thread(target=_probe, name="gpu-probe", daemon=True).start()


def available() -> bool:
    return bool(_probe())


def backend_name() -> str:
    """Which path ran the last batch, for the HUD."""
    return _last_backend


def _forced() -> bool:
    return _env("BEAM_PROFILER_GPU") in ("1", "on", "yes", "force", "true")


def fit_batch(img, x0, y0, cw, ch):
    """Fit all crops in one kernel launch.

    Returns (indices that produced a fit, SpotArray) or None when the GPU path
    does not apply, in which case the caller uses numpy."""
    global _last_backend
    _last_backend = "cpu"
    n = len(x0)
    if n < MIN_SPOTS and not _forced():
        return None
    cp = _probe()
    if not cp:
        return None
    entry = _ENTRY.get(img.dtype)
    if entry is None or int(cw.max()) > MAX_CROP or int(ch.max()) > MAX_CROP:
        return None
    try:
        frame = _upload(cp, img)
        d_out = cp.empty((n, 9), dtype=cp.float64)
        d_out[:, 8] = 0.0
        _kernels[entry](
            (n,), (BLOCK,),
            (frame, np.int32(img.shape[1]),
             cp.asarray(x0, dtype=cp.int64), cp.asarray(y0, dtype=cp.int64),
             cp.asarray(cw, dtype=cp.int32), cp.asarray(ch, dtype=cp.int32),
             d_out, np.int32(n)),
        )
        out = cp.asnumpy(d_out)
    except Exception as e:
        logging.warning("GPU spot fitting failed (%s); falling back to the CPU path", e)
        globals()["_cupy"] = False
        return None

    _last_backend = "cuda"
    keep = np.nonzero(out[:, 8] > 0)[0]
    o = out[keep]
    spots = SpotArray(
        o[:, 0], o[:, 1], o[:, 2], o[:, 3],
        np.stack([o[:, 4], o[:, 5]], axis=1),
        np.stack([-o[:, 5], o[:, 4]], axis=1),
        o[:, 6], o[:, 7],
    )
    return keep, spots


def _upload(cp, img):
    """Copy the frame into a reused device buffer (avoids per-frame allocation)."""
    global _frame_buf
    if _frame_buf is None or _frame_buf.shape != img.shape or _frame_buf.dtype != img.dtype:
        _frame_buf = cp.empty(img.shape, dtype=img.dtype)
    _frame_buf.set(np.ascontiguousarray(img))
    return _frame_buf
