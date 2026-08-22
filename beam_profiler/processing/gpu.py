"""Optional CuPy backend for the batched spot fit.

Everything here is best-effort: if CuPy, a driver or a usable device is
missing, `fit_batch` returns None and the caller falls back to numpy.  Import
of this module never imports CuPy - the probe happens on first use so the
CUDA context cost (~0.8 s) is only paid by machines that have one.

Why a hand-written kernel instead of stacked CuPy array ops: the per-spot work
is tiny (a ~35x35 crop), so the GPU spends its time waiting for Python to issue
the next launch.  The array-op version needs ~50 launches per distinct crop
size *per stage*; the kernel below runs every stage, every adaptive iteration
and every crop size in a **single launch**, one block per spot, with each block
resizing its own crop between stages.

The maths must stay identical to `fit.fit_spot` - `tests/test_processing.py`
asserts that to 1e-9, so any change here has to be mirrored there and in
`fit.fit_crops`.

Set BEAM_PROFILER_GPU=0 to force the CPU path, or =force to use the GPU even
below the spot-count threshold (useful for benchmarking).  Without the
variable the settings window (`gpu_enabled`, `gpu_min_spots`) decides.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

from .spots import SpotArray

MAX_CROP = 512  # kernel shared-memory limit on the crop side, px
BLOCK = 128  # threads per block, must be a power of two

_cupy = None  # None = not probed yet, False = unusable
_kernels: dict = {}
_frame_buf = None
_probe_lock = threading.Lock()
_last_backend = "cpu"  # which path actually ran the last batch, for the HUD

_KERNEL_SOURCE = r"""
#define BLOCK %(block)d
#define MAX_BORDER %(max_border)d
#define I0W_NORM %(i0w_norm).17g
#define QUAD_MAX %(quad_max).17g
#define EPSV %(eps).17g

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

// index of the sample nearest to c, ties to the lower index (numpy argmin)
__device__ __forceinline__ int nearest_index(double c, int n)
{
    int lo = (int)floor(c);
    if (lo < 0) return 0;
    if (lo >= n - 1) return n - 1;
    return (fabs(c - (double)lo) <= fabs(c - (double)(lo + 1))) ? lo : lo + 1;
}

__device__ __forceinline__ int quantise(double twice_half, int q, int lo, int hi)
{
    int i = (int)(ceil(twice_half / (double)q) * (double)q);
    return i < lo ? lo : (i > hi ? hi : i);
}

#define FIT_KERNEL(NAME, PIXEL)                                                     \
extern "C" __global__ void NAME(                                                    \
    const PIXEL *__restrict__ img, const int W, const int H,                        \
    const double *__restrict__ cx0, const double *__restrict__ cy0,                 \
    const double *__restrict__ hx0, const double *__restrict__ hy0,                 \
    const double *__restrict__ capa,                                                \
    const double crop_sigma, const int stages, const int adaptive_iters,            \
    const int quantum, const int min_crop, const int max_crop,                      \
    const int subtract_bg, const int clip_neg,                                      \
    double *__restrict__ out, const int n)                                          \
{                                                                                   \
    const int b = blockIdx.x;                                                       \
    if (b >= n) return;                                                             \
    const int t = threadIdx.x;                                                      \
                                                                                    \
    __shared__ double sh[BLOCK];                                                    \
    __shared__ float border[MAX_BORDER];                                            \
    __shared__ float med_lo, med_hi;                                                \
                                                                                    \
    double cx = cx0[b], cy = cy0[b];                                                \
    double hx = hx0[b], hy = hy0[b];                                                \
    const double cap = capa[b];                                                     \
    double cxx = 0.0, cyy = 0.0, cxy = 0.0;                                         \
    int x0 = 0, y0 = 0, cw = 0, ch = 0;                                             \
    int px0 = -1, py0 = -1, pcw = -1, pch = -1;                                     \
    float bg = 0.0f;                                                                \
                                                                                    \
    const int hi_w = max_crop < W ? max_crop : W;                                   \
    const int hi_h = max_crop < H ? max_crop : H;                                   \
                                                                                    \
    for (int stage = 0; stage <= stages; ++stage) {                                 \
        cw = quantise(2.0 * hx, quantum, min_crop, hi_w);                           \
        ch = quantise(2.0 * hy, quantum, min_crop, hi_h);                           \
        x0 = (int)rint(cx) - cw / 2;   /* rint: half-to-even, like np.round */      \
        y0 = (int)rint(cy) - ch / 2;                                                \
        if (x0 < 0) x0 = 0;                                                         \
        if (x0 > W - cw) x0 = W - cw;                                               \
        if (y0 < 0) y0 = 0;                                                         \
        if (y0 > H - ch) y0 = H - ch;                                               \
        /* the fit depends only on the crop, so a crop that did not move would   */ \
        /* return exactly what it returned last stage.  The final stage always   */ \
        /* runs, because only it computes I0 and the ellipse mean.               */ \
        const int settled = (cw == pcw && ch == pch && x0 == px0 && y0 == py0);     \
        pcw = cw; pch = ch; px0 = x0; py0 = y0;                                     \
        if (settled && stage < stages) continue;                                    \
        const int npix = cw * ch;                                                   \
                                                                                    \
        /* ---- local background: median of the crop border ---- */                 \
        bg = 0.0f;                                                                  \
        if (subtract_bg) {                                                          \
            const int m = 2 * (cw + ch);  /* corners twice, as np.concatenate */    \
            for (int i = t; i < m; i += BLOCK) {                                    \
                float v;                                                            \
                if (i < cw)               v = img[(long long)y0 * W + x0 + i];      \
                else if (i < 2 * cw)      v = img[(long long)(y0 + ch - 1) * W + x0 + i - cw]; \
                else if (i < 2 * cw + ch) v = img[(long long)(y0 + i - 2 * cw) * W + x0]; \
                else                      v = img[(long long)(y0 + i - 2 * cw - ch) * W + x0 + cw - 1]; \
                border[i] = v;                                                      \
            }                                                                       \
            if (t == 0) { med_lo = 0.0f; med_hi = 0.0f; }                           \
            __syncthreads();                                                        \
            const int k_lo = (m - 1) / 2, k_hi = m / 2;                             \
            for (int i = t; i < m; i += BLOCK) {                                    \
                const float v = border[i];                                          \
                int lt = 0, eq = 0;                                                 \
                for (int j = 0; j < m; ++j) { float u = border[j]; lt += (u < v); eq += (u == v); } \
                if (lt <= k_lo && k_lo < lt + eq) med_lo = v;                       \
                if (lt <= k_hi && k_hi < lt + eq) med_hi = v;                       \
            }                                                                       \
            __syncthreads();                                                        \
            bg = (m & 1) ? med_lo : (med_lo + med_hi) * 0.5f;                       \
        }                                                                           \
                                                                                    \
        /* ---- plain second moments ---- */                                        \
        double S = 0.0, Sx = 0.0, Sy = 0.0;                                         \
        for (int p = t; p < npix; p += BLOCK) {                                     \
            const int r = p / cw, c = p - r * cw;                                   \
            float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;            \
            if (clip_neg && z < 0.0f) z = 0.0f;                                     \
            const double zd = (double)z;                                            \
            S += zd; Sx += zd * (double)(x0 + c); Sy += zd * (double)(y0 + r);      \
        }                                                                           \
        S = block_sum(S, sh); Sx = block_sum(Sx, sh); Sy = block_sum(Sy, sh);       \
        if (!(S > 0.0)) { if (t == 0) out[b * 9 + 8] = 0.0; return; }               \
        cx = Sx / S; cy = Sy / S;                                                   \
                                                                                    \
        double axx = 0.0, ayy = 0.0, axy = 0.0;                                     \
        for (int p = t; p < npix; p += BLOCK) {                                     \
            const int r = p / cw, c = p - r * cw;                                   \
            float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;            \
            if (clip_neg && z < 0.0f) z = 0.0f;                                     \
            const double zd = (double)z;                                            \
            const double dx = (double)(x0 + c) - cx, dy = (double)(y0 + r) - cy;    \
            axx += zd * dx * dx; ayy += zd * dy * dy; axy += zd * dx * dy;          \
        }                                                                           \
        cxx = block_sum(axx, sh) / S;                                               \
        cyy = block_sum(ayy, sh) / S;                                               \
        cxy = block_sum(axy, sh) / S;                                               \
                                                                                    \
        /* ---- adaptive: Gaussian weight matched to C, fixed point C = 2M ---- */  \
        for (int it = 0; it < adaptive_iters; ++it) {                               \
            const double det = cxx * cyy - cxy * cxy;                               \
            if (!(det > 0.0)) break;                                                \
            const double ia = cyy / det, ib = cxy / det, ic = cxx / det;            \
            double ws = 0.0, wxs = 0.0, wys = 0.0;                                  \
            for (int p = t; p < npix; p += BLOCK) {                                 \
                const int r = p / cw, c = p - r * cw;                               \
                float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;        \
                if (clip_neg && z < 0.0f) z = 0.0f;                                 \
                const double dx = (double)(x0 + c) - cx, dy = (double)(y0 + r) - cy; \
                double q = ia * dx * dx + ic * dy * dy - 2.0 * ib * dy * dx;        \
                q = q < 0.0 ? 0.0 : (q > QUAD_MAX ? QUAD_MAX : q);                  \
                const double wz = exp(-0.5 * q) * (double)z;                        \
                ws += wz; wxs += wz * (double)(x0 + c); wys += wz * (double)(y0 + r); \
            }                                                                       \
            ws = block_sum(ws, sh); wxs = block_sum(wxs, sh); wys = block_sum(wys, sh); \
            if (!(ws > 0.0)) break;                                                 \
            const double nx = wxs / ws, ny = wys / ws;                              \
            double bxx = 0.0, byy = 0.0, bxy = 0.0;                                 \
            for (int p = t; p < npix; p += BLOCK) {                                 \
                const int r = p / cw, c = p - r * cw;                               \
                float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;        \
                if (clip_neg && z < 0.0f) z = 0.0f;                                 \
                const double dx = (double)(x0 + c) - cx, dy = (double)(y0 + r) - cy; \
                double q = ia * dx * dx + ic * dy * dy - 2.0 * ib * dy * dx;        \
                q = q < 0.0 ? 0.0 : (q > QUAD_MAX ? QUAD_MAX : q);                  \
                const double wz = exp(-0.5 * q) * (double)z;                        \
                const double ex = (double)(x0 + c) - nx, ey = (double)(y0 + r) - ny; \
                bxx += wz * ex * ex; byy += wz * ey * ey; bxy += wz * ex * ey;      \
            }                                                                       \
            bxx = block_sum(bxx, sh); byy = block_sum(byy, sh); bxy = block_sum(bxy, sh); \
            cx = nx; cy = ny;                                                       \
            cxx = 2.0 * bxx / ws; cyy = 2.0 * byy / ws; cxy = 2.0 * bxy / ws;       \
        }                                                                           \
                                                                                    \
        if (stage < stages) {                                                       \
            const double sx = crop_sigma * sqrt(cxx > EPSV ? cxx : EPSV);           \
            const double sy = crop_sigma * sqrt(cyy > EPSV ? cyy : EPSV);           \
            hx = sx < cap ? sx : cap;                                               \
            hy = sy < cap ? sy : cap;                                               \
        }                                                                           \
    }                                                                               \
                                                                                    \
    /* ---- peak intensity and the 1-sigma ellipse mean, on the final crop ---- */  \
    const double det = cxx * cyy - cxy * cxy;                                       \
    double e_sum = 0.0, e_cnt = 0.0;                                                \
    if (det > 0.0) {                                                                \
        const double ia = cyy / det, ib = cxy / det, ic = cxx / det;                \
        for (int p = t; p < cw * ch; p += BLOCK) {                                  \
            const int r = p / cw, c = p - r * cw;                                   \
            float z = (float)img[(long long)(y0 + r) * W + x0 + c] - bg;            \
            if (clip_neg && z < 0.0f) z = 0.0f;                                     \
            const double dx = (double)(x0 + c) - cx, dy = (double)(y0 + r) - cy;    \
            if (ia * dx * dx + ic * dy * dy - 2.0 * ib * dy * dx <= 1.0) {          \
                e_sum += (double)z; e_cnt += 1.0;                                   \
            }                                                                       \
        }                                                                           \
        e_sum = block_sum(e_sum, sh);                                               \
        e_cnt = block_sum(e_cnt, sh);                                               \
    }                                                                               \
    if (t != 0) return;                                                             \
    const int col = nearest_index(cx - (double)x0, cw);                             \
    const int row = nearest_index(cy - (double)y0, ch);                             \
    float zc = (float)img[(long long)(y0 + row) * W + x0 + col] - bg;               \
    if (clip_neg && zc < 0.0f) zc = 0.0f;                                           \
    const double I0 = (double)zc;                                                   \
    const double I0w = (det > 0.0 && e_cnt > 0.0) ? e_sum / (e_cnt * I0W_NORM) : I0; \
                                                                                    \
    const double half_tr = 0.5 * (cxx + cyy);                                       \
    double disc = 0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy;                     \
    disc = disc > 0.0 ? sqrt(disc) : 0.0;                                           \
    const double l_max = half_tr + disc, l_min = half_tr - disc;                    \
    double vx = cxy, vy = l_max - cxx;                                              \
    const double vn = sqrt(vx * vx + vy * vy);                                      \
    if (vn < EPSV) { vx = 1.0; vy = 0.0; } else { vx /= vn; vy /= vn; }             \
                                                                                    \
    out[b * 9 + 0] = cx;                                                            \
    out[b * 9 + 1] = cy;                                                            \
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
        from .fit import _EPS, _I0W_NORM, _QUAD_MAX  # noqa: PLC0415

        source = _KERNEL_SOURCE % {
            "block": BLOCK,
            "max_border": 4 * MAX_CROP,
            "i0w_norm": _I0W_NORM,
            "quad_max": _QUAD_MAX,
            "eps": _EPS,
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


def fit_batch(img, x, y, r, cap, cfg):
    """Run every crop stage and adaptive iteration in one kernel launch.

    Returns (indices that produced a fit, SpotArray) or None when the GPU path
    does not apply, in which case the caller uses numpy."""
    global _last_backend
    _last_backend = "cpu"
    n = len(x)
    if not cfg.gpu_enabled and not _forced():
        return None
    if n < cfg.gpu_min_spots and not _forced():
        return None
    if cfg.max_crop > MAX_CROP:  # the kernel would clamp differently than numpy
        return None
    cp = _probe()
    if not cp:
        return None
    entry = _ENTRY.get(img.dtype)
    if entry is None:
        return None
    try:
        frame = _upload(cp, img)
        d_out = cp.zeros((n, 9), dtype=cp.float64)
        half = np.minimum(cfg.seed_window * r, cap)
        d = lambda v: cp.asarray(np.ascontiguousarray(v, dtype=np.float64))  # noqa: E731
        H, W = img.shape
        _kernels[entry](
            (n,), (BLOCK,),
            (frame, np.int32(W), np.int32(H),
             d(x), d(y), d(half), d(half),
             d(np.where(np.isfinite(cap), cap, 1e18)),
             np.float64(cfg.crop_sigma), np.int32(cfg.crop_stages),
             np.int32(cfg.adaptive_iters), np.int32(max(1, cfg.crop_quantum)),
             np.int32(cfg.min_crop), np.int32(cfg.max_crop),
             np.int32(bool(cfg.subtract_background)), np.int32(bool(cfg.clip_negative)),
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
