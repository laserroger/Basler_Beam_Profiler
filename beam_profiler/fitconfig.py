"""Tunable parameters of the spot fit, plus their live-editable registry.

Every number the estimator depends on lives here rather than as a literal in
`processing/fit.py`, so it can be changed from the settings window (press `o`
in the viewer), over the HTTP API, or in `fit_config.json` next to the app.

The defaults are not guesses - each one is the measured optimum, and
`tests/test_fit_config.py` re-runs the measurement so a default cannot drift
without a test failing.  `docs/fitting.md` explains the method step by step;
`docs/fitting-calibration.md` records the sweeps behind each default.

Readers must call `active()` once per frame and use that snapshot: the
settings window swaps in a whole new FitConfig object rather than mutating the
live one, so a frame can never see a half-applied change.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields, replace

from .config import APP_DIR

CONFIG_PATH = os.path.join(APP_DIR, "fit_config.json")


@dataclass(frozen=True)
class FitConfig:
    # --- what gets measured ------------------------------------------------
    crop_sigma: float = 3.5
    crop_stages: int = 3
    adaptive_iters: int = 0
    seed_window: float = 2.0
    subtract_background: bool = True
    clip_negative: bool = False
    # --- crop bounds -------------------------------------------------------
    neighbour_fraction: float = 0.5
    crop_quantum: int = 4
    min_crop: int = 3
    max_crop: int = 512
    # --- backend -----------------------------------------------------------
    gpu_enabled: bool = True
    gpu_min_spots: int = 400

    def replace(self, **kw) -> FitConfig:
        return replace(self, **{k: v for k, v in kw.items() if k in _NAMES})


@dataclass(frozen=True)
class Setting:
    """UI metadata for one FitConfig field."""

    name: str
    label: str
    kind: str  # "float" | "int" | "bool"
    lo: float
    hi: float
    group: str
    help: str


SETTINGS: tuple[Setting, ...] = (
    Setting(
        "crop_sigma", "Crop half-width (x sigma)", "float", 1.5, 12.0, "Estimator",
        "Crop half-width per axis, in units of the fitted sigma of that axis.\n"
        "Measured optimum 3.5 (min RMS error: 1.5% bright, 2.1% faint).\n"
        "Below 3 the width is truncated; above 4.5 tail noise takes over.\n"
        "With adaptive iterations > 0 the result is flat over 3.5-8, so the\n"
        "exact value stops mattering.",
    ),
    Setting(
        "crop_stages", "Re-crop stages", "int", 0, 8, "Estimator",
        "How many times the crop is resized from the fitted sigma and the fit\n"
        "repeated.  The blob detector's radius estimate spans r/sigma = 0.75-1.7,\n"
        "and 3 stages converge from the worst of those to <0.1%.  0 disables\n"
        "re-cropping and fits the blob-radius crop directly (old behaviour).",
    ),
    Setting(
        "adaptive_iters", "Adaptive weight iterations", "int", 0, 20, "Estimator",
        "0 = plain second moments (ISO 11146 D4sigma style, fastest).\n"
        ">0 = iterate a Gaussian weight matched to the spot (C = 2M at the\n"
        "fixed point).  Removes the window dependence entirely: bias and\n"
        "spread stay ~0.1% for any crop from 3.5 to 8 sigma, and faint spots\n"
        "get ~8x better per-frame precision.  Converges by ~0.5x per iteration:\n"
        "4 iters -> 0.6%, 6 -> 0.2%, 8 -> 0.1%.  Costs one extra pass over the\n"
        "crop each, so ~10x the fit time at 8.",
    ),
    Setting(
        "seed_window", "Seed crop (x blob radius)", "float", 1.0, 6.0, "Estimator",
        "First crop, before any sigma is known, as a multiple of the blob\n"
        "detector's radius.  Only has to be good enough to start the re-crop\n"
        "stages; a larger value converges in fewer stages.",
    ),
    Setting(
        "subtract_background", "Subtract local background", "bool", 0, 1, "Estimator",
        "Subtract the median of the crop border from the crop.  Without it a\n"
        "constant pedestal inflates the second moments badly.",
    ),
    Setting(
        "clip_negative", "Clip negatives to zero", "bool", 0, 1, "Estimator",
        "OFF by default, and should stay off.  Clipping rectifies zero-mean\n"
        "noise into a positive pedestal (E[max(z,0)] = 0.4*sigma_noise) which\n"
        "the second moment then multiplies by distance^2.  The error grows\n"
        "without bound with crop size - at 12 sigma it reads +160% (bright)\n"
        "and +364% (faint), converging on the width of the crop itself rather\n"
        "than the beam.  Provided only to reproduce pre-2026 numbers.",
    ),
    Setting(
        "neighbour_fraction", "Max crop / neighbour distance", "float", 0.1, 1.0, "Crop limits",
        "Crop half-width is capped at this fraction of the distance to the\n"
        "nearest other candidate, so a crop cannot swallow its neighbour.\n"
        "0.5 = stop at the midpoint between two spots.",
    ),
    Setting(
        "crop_quantum", "Crop size quantum (px)", "int", 1, 32, "Crop limits",
        "Crop dimensions are rounded up to a multiple of this.  Purely a speed\n"
        "knob for the CPU path, which needs one tensor per distinct crop size:\n"
        "coarser quantisation means fewer, larger batches.  Ignored by the GPU\n"
        "kernel, which handles a different size per spot.",
    ),
    Setting(
        "min_crop", "Minimum crop (px)", "int", 3, 64, "Crop limits",
        "Candidates whose crop is smaller than this in either axis are dropped.",
    ),
    Setting(
        "max_crop", "Maximum crop (px)", "int", 16, 4096, "Crop limits",
        "Hard upper bound on crop size, so a runaway fit cannot try to read the\n"
        "whole sensor.  512 is also the CUDA kernel's limit: raising this above\n"
        "512 silently moves the whole fit to the (much slower) CPU path.  Only\n"
        "needed for beams wider than about 70 px sigma.",
    ),
    Setting(
        "gpu_enabled", "Use CUDA when available", "bool", 0, 1, "Backend",
        "Run the batched fit as a single fused CUDA kernel when CuPy and a\n"
        "device are present.  Falls back to numpy automatically otherwise.\n"
        "The BEAM_PROFILER_GPU environment variable overrides this.",
    ),
    Setting(
        "gpu_min_spots", "Minimum spots for CUDA", "int", 1, 100_000, "Backend",
        "Below this many candidates the frame upload and launch cost more than\n"
        "the fit saves, so the CPU path is used.  Measured crossover ~300.",
    ),
)

_NAMES = {f.name for f in fields(FitConfig)}
assert {s.name for s in SETTINGS} == _NAMES, "SETTINGS and FitConfig are out of sync"

_active = FitConfig()
_listeners: list = []


def active() -> FitConfig:
    """The config in force right now.  Snapshot it once per frame."""
    return _active


def set_active(cfg: FitConfig, save: bool = True):
    """Swap in a new config atomically and notify listeners."""
    global _active
    _active = clamp(cfg)
    if save:
        save_config(_active)
    for fn in list(_listeners):
        try:
            fn(_active)
        except Exception as e:
            logging.error(f"fit config listener failed: {e}")


def on_change(fn):
    """Register a callback invoked with the new config after every change."""
    _listeners.append(fn)
    return fn


def clamp(cfg: FitConfig) -> FitConfig:
    """Force every field into its declared range and type."""
    out = {}
    for s in SETTINGS:
        v = getattr(cfg, s.name)
        if s.kind == "bool":
            out[s.name] = bool(v)
        elif s.kind == "int":
            out[s.name] = int(min(max(int(v), s.lo), s.hi))
        else:
            out[s.name] = float(min(max(float(v), s.lo), s.hi))
    return FitConfig(**out)


def load_config(path: str | None = None) -> FitConfig:
    """Read fit_config.json; unknown or malformed entries fall back to defaults."""
    path = path or CONFIG_PATH  # resolved per call, so the path stays overridable
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in _NAMES}
        if len(known) != len(data):
            logging.warning(f"ignoring unknown keys in {path}: {set(data) - _NAMES}")
        return clamp(FitConfig(**known))
    except FileNotFoundError:
        return FitConfig()
    except Exception as e:
        logging.error(f"could not read {path} ({e}); using defaults")
        return FitConfig()


def save_config(cfg: FitConfig, path: str | None = None):
    path = path or CONFIG_PATH
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=4)
    except Exception as e:
        logging.error(f"could not write {path}: {e}")


def describe(cfg: FitConfig | None = None) -> str:
    """One-line summary for the HUD."""
    cfg = cfg or active()
    mode = f"adaptive x{cfg.adaptive_iters}" if cfg.adaptive_iters else "moments"
    clip = ", clipped" if cfg.clip_negative else ""
    return f"{mode}, crop {cfg.crop_sigma:g}s x{cfg.crop_stages}{clip}"


set_active(load_config(), save=False)
