"""CV pipeline tests against synthetic images with known ground truth."""

import numpy as np
import pytest

from beam_profiler.processing import classify_grid, detect_spots, region_stats
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
    assert detect_spots(img) == []


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
