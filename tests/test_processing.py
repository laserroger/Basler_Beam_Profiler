"""CV pipeline tests against synthetic images with known ground truth."""

import numpy as np
import pytest

from beam_profiler.fitconfig import FitConfig
from beam_profiler.processing import (
    SpotArray,
    classify_grid,
    detect_spots,
    fit,
    fit_spot,
    fit_spots,
    gpu,
    region_stats,
)
from beam_profiler.processing.blobs import _DETECTOR, candidate_mask
from beam_profiler.synthetic import Spot, render, spot_grid


def _match(spots, truth, tol=3.0):
    """Pair each ground-truth spot with the nearest detection within tol px."""
    pairs = []
    for t in truth:
        best = min(spots, key=lambda s: np.hypot(s["x"] - t.x, s["y"] - t.y))
        assert np.hypot(best["x"] - t.x, best["y"] - t.y) < tol, (
            f"no detection near ({t.x}, {t.y})"
        )
        pairs.append((best, t))
    return pairs


def test_single_spot_position_and_width():
    truth = [Spot(600.5, 420.25, sigma_x=15.0, amplitude=0.8)]
    img = render(1200, 900, truth, rng=np.random.default_rng(1))
    spots = detect_spots(img)
    assert len(spots) == 1
    ((s, t),) = _match(spots, truth, tol=1.0)
    # detection convention: sigma_0/1 = 2x the Gaussian sigma
    assert s["sigma_0"] == pytest.approx(2 * t.sigma_x, rel=0.15)
    assert s["sigma_1"] == pytest.approx(2 * t.sigma_x, rel=0.15)


def test_elliptical_spot_orientation():
    theta = np.radians(30)
    truth = [Spot(600, 600, sigma_x=30.0, sigma_y=12.0, theta=theta, amplitude=0.8)]
    img = render(1200, 1200, truth, rng=np.random.default_rng(2))
    spots = detect_spots(img)
    assert len(spots) == 1
    s = spots[0]
    assert s["sigma_0"] > s["sigma_1"]
    assert s["sigma_0"] == pytest.approx(2 * 30.0, rel=0.15)
    assert s["sigma_1"] == pytest.approx(2 * 12.0, rel=0.2)
    angle = np.arctan2(s["vec_0"][1], s["vec_0"][0]) % np.pi
    assert angle == pytest.approx(theta, abs=np.radians(5))


def test_grid_detection_and_classification():
    truth = spot_grid(1200, 1200, 4, 4, sigma=10.0)
    img = render(1200, 1200, truth, rng=np.random.default_rng(3))
    spots = detect_spots(img)
    assert len(spots) == 16
    _match(spots, truth, tol=1.5)

    rows, columns, stats = classify_grid(spots, eps=20)
    assert len(rows) == 4 and all(len(r) == 4 for r in rows)
    assert len(columns) == 4 and all(len(c) == 4 for c in columns)
    pitch = truth[1].x - truth[0].x
    assert stats["rows"]["avg_mean_dx"] == pytest.approx(pitch, rel=0.02)
    assert stats["columns"]["avg_mean_dy"] == pytest.approx(pitch, rel=0.02)
    assert stats["rows"]["avg_std_y"] < 1.0  # rows are straight


def test_empty_frame_yields_no_spots():
    img = render(800, 800, [], noise=0.003, rng=np.random.default_rng(4))
    assert len(detect_spots(img)) == 0


def test_region_stats():
    img = np.zeros((100, 100), dtype=np.uint16)
    img[20:40, 10:30] = 1000
    roi = (100, 100, 50, 50)  # frame grabbed at sensor offset (50, 50)
    stats = region_stats(img, roi, (60, 70, 80, 90), pixel_size=2.5e-6)
    assert stats["width_pixels"] == 20 and stats["height_pixels"] == 20
    assert stats["roi_coords"] == [10, 20, 30, 40]
    assert stats["max"] == 1000 and stats["sum"] == 1000 * 400
    assert stats["width_um"] == pytest.approx(50.0)
    # rectangle entirely outside the ROI
    assert region_stats(img, roi, (0, 0, 10, 10), pixel_size=2.5e-6) is None


def test_simulated_camera_pipeline():
    from beam_profiler.cameras import SimulatedCamera

    cam = SimulatedCamera(grid=(3, 3), jitter=0.0)
    frame = cam.grab_image()
    assert frame.shape == (1200, 1200) and frame.dtype == np.uint16
    assert len(detect_spots(frame)) == 9
    # ROI crop keeps only the spots inside
    cam.ROI = (400, 400, 0, 0)
    frame = cam.grab_image()
    assert frame.shape == (400, 400)
    # exposure scales intensity
    lo = cam
    lo.ROI = (1200, 1200, 0, 0)
    lo.ExposureTime = 50
    dim = lo.grab_image().max()
    lo.ExposureTime = 400
    bright = lo.grab_image().max()
    assert bright > dim



# --------------------------------------------------------------------------- #
#  the batched paths must stay numerically identical to the scalar reference
# --------------------------------------------------------------------------- #
def _candidates(img):
    """Blob-detector output, i.e. exactly what the fit is handed."""
    mask, scale = candidate_mask(img)
    keypoints = _DETECTOR.detect(mask)
    pts = np.array([(k.pt[0], k.pt[1], k.size) for k in keypoints])
    return pts[:, 0] / scale, pts[:, 1] / scale, np.maximum(pts[:, 2] / (2 * scale), 2.0)


def _dense_frame(n=12, sigma=4.0, size=1200, seed=5):
    truth = spot_grid(size, size, n, n, sigma=sigma)
    return render(size, size, truth, rng=np.random.default_rng(seed)), truth


def _reference(img, x, y, r, cfg):
    """Scalar fit_spot over the same candidates, with the same neighbour caps."""
    cap = fit.neighbour_limit(x, y, cfg)
    fits = (fit_spot(img, a, b, c, cfg, k) for a, b, c, k in zip(x, y, r, cap))
    return [s for s in fits if s is not None]


def _assert_same(reference, batched, tol=1e-9):
    assert len(reference) == len(batched)
    peak = max(abs(s["I0"]) for s in reference) or 1.0
    for a, b in zip(reference, batched):
        for key in ("x", "y", "sigma_0", "sigma_1"):
            assert b[key] == pytest.approx(a[key], abs=tol), key
        assert b["I0"] == pytest.approx(a["I0"], abs=tol * peak)
        assert b["I0_weighted"] == pytest.approx(a["I0_weighted"], abs=tol * peak)
        # the major axis is ill-conditioned on a round spot, so only check
        # orientation where there is a meaningful axis to check
        if a["sigma_0"] > 1.02 * a["sigma_1"]:
            assert abs(abs(float(np.dot(a["vec_0"], b["vec_0"]))) - 1.0) < tol


CONFIGS = {
    "default": FitConfig(),
    "adaptive": FitConfig(adaptive_iters=6),
    "no re-crop": FitConfig(crop_stages=0),
    "clip on (legacy)": FitConfig(clip_negative=True),
    "coarse quantum": FitConfig(crop_quantum=16, crop_sigma=4.5),
}


@pytest.mark.parametrize("label", list(CONFIGS))
def test_batched_fit_matches_scalar_reference(label, monkeypatch):
    cfg = CONFIGS[label]
    monkeypatch.setenv("BEAM_PROFILER_GPU", "0")  # force the numpy path
    monkeypatch.setattr(gpu, "_cupy", False)
    img, _ = _dense_frame()
    x, y, r = _candidates(img)
    _assert_same(_reference(img, x, y, r, cfg), fit_spots(img, x, y, r, cfg))


@pytest.mark.skipif(not gpu.available(), reason="no CUDA device / CuPy")
@pytest.mark.parametrize("label", list(CONFIGS))
def test_cuda_fit_matches_scalar_reference(label, monkeypatch):
    cfg = CONFIGS[label]
    monkeypatch.setenv("BEAM_PROFILER_GPU", "force")  # force the GPU path
    img, _ = _dense_frame()
    x, y, r = _candidates(img)
    got = fit_spots(img, x, y, r, cfg)
    assert gpu.backend_name() == "cuda", "GPU path did not engage"
    _assert_same(_reference(img, x, y, r, cfg), got)


def test_recrop_recovers_from_a_bad_seed_radius():
    """The blob detector's radius varies with brightness; the re-crop stages
    must converge to the same width regardless of where they start."""
    truth = [Spot(600.0, 600.0, sigma_x=8.0, amplitude=0.8)]
    img = render(1200, 1200, truth, rng=np.random.default_rng(6))
    widths = [
        fit_spot(img, 600.0, 600.0, 8.0 * factor)["sigma_0"] for factor in (0.75, 1.0, 1.5, 2.0)
    ]
    assert max(widths) - min(widths) < 0.02 * np.mean(widths)
    assert np.mean(widths) == pytest.approx(2 * 8.0, rel=0.03)


def test_large_spots_are_fitted_at_full_resolution():
    """No coarse-graining: a big spot must not read wider than a small one."""
    for sigma in (6.0, 20.0, 45.0):
        img = render(1600, 1600, [Spot(800.0, 800.0, sigma_x=sigma, amplitude=0.8)],
                     rng=np.random.default_rng(7))
        spots = detect_spots(img)
        assert len(spots) == 1
        assert spots.sigma_0[0] == pytest.approx(2 * sigma, rel=0.04)


def test_crop_never_swallows_a_neighbour():
    cfg = FitConfig(crop_sigma=8.0)  # deliberately greedy
    truth = spot_grid(1200, 1200, 6, 6, sigma=5.0)
    img = render(1200, 1200, truth, rng=np.random.default_rng(8))
    x, y, r = _candidates(img)
    cap = fit.neighbour_limit(x, y, cfg)
    pitch = truth[1].x - truth[0].x
    assert np.all(cap <= 0.5 * pitch * 1.001)
    spots = fit_spots(img, x, y, r, cfg)
    assert len(spots) == 36
    assert np.median(spots.sigma_0) == pytest.approx(2 * 5.0, rel=0.15)


def test_spot_array_views():
    img, truth = _dense_frame()
    spots = detect_spots(img)
    assert isinstance(spots, SpotArray)
    assert len(spots) == len(truth)
    assert spots.sigma[0] == pytest.approx(np.sqrt(spots.sigma_0[0] * spots.sigma_1[0]))
    assert set(spots[0]) == {"x", "y", "sigma_0", "sigma_1", "sigma", "vec_0", "vec_1",
                             "I0", "I0_weighted"}
    assert spots[0]["x"] == spots.x[0]
    subset = spots[spots.x < 600]
    assert len(subset) < len(spots) and np.all(subset.x < 600)
    assert len(spots.to_json()) == len(spots)
    assert spots.to_json()[0]["vec_0"] == list(spots.vec_0[0])
