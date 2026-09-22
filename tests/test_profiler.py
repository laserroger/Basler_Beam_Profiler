from types import SimpleNamespace

import numpy as np
import pytest

from beam_profiler.processing.profiler import fit_single_beam
from beam_profiler.processing.spots import SpotArray
from beam_profiler.ui.viewer import Viewer


def beam(h=768, w=1024, cx=500, cy=390, sx=220, sy=140, angle=0.4):
    y, x = np.mgrid[:h, :w]
    u = np.cos(angle) * (x-cx) + np.sin(angle) * (y-cy)
    v = -np.sin(angle) * (x-cx) + np.cos(angle) * (y-cy)
    return (3000 + 35000*np.exp(-0.5*((u/sx)**2+(v/sy)**2))).astype(np.uint16)


def test_broad_rotated_beam_with_illuminated_borders():
    result = fit_single_beam(beam())
    spot = result.spots[0]
    assert spot['x'] == pytest.approx(500, abs=0.5)
    assert spot['y'] == pytest.approx(390, abs=0.5)
    assert spot['sigma_0'] == pytest.approx(440, rel=0.005)
    assert spot['sigma_1'] == pytest.approx(280, rel=0.005)
    assert abs(np.dot(spot['vec_0'], [np.cos(.4), np.sin(.4)])) > 0.999


def test_noise_and_local_dark_defect_do_not_become_separate_spots():
    image = beam().astype(float)
    image[260:280, 420:440] *= .3
    image += np.random.default_rng(4).normal(0, 150, image.shape)
    result = fit_single_beam(image.astype(np.uint16))
    assert len(result.spots) == 1
    assert result.spots.sigma_0[0] == pytest.approx(440, rel=.015)


def test_partial_beam_reports_edge_estimate():
    result = fit_single_beam(beam(cx=160))
    assert len(result.spots) == 1
    assert 'outside region' in result.message


def test_zero_background_is_valid():
    image = beam().astype(np.int32) - 3000
    result = fit_single_beam(image.astype(np.uint16))
    assert len(result.spots) == 1
    assert result.spots.sigma_0[0] == pytest.approx(440, rel=.005)


def test_optimizer_cannot_explain_image_with_negative_background(monkeypatch):
    from beam_profiler.processing import profiler
    optimize = profiler.least_squares
    backgrounds = []
    image = beam()
    small = profiler.cv2.resize(image.astype(np.float32), (128, 96), interpolation=profiler.cv2.INTER_AREA)
    low, high = np.percentile(small, [1, 99])
    def checked(*args, **kwargs):
        result = optimize(*args, **kwargs)
        backgrounds.append(result.x[0] * (high-low) + low)
        return result
    monkeypatch.setattr(profiler, 'least_squares', checked)
    profiler.fit_single_beam(image)
    assert backgrounds and backgrounds[0] >= -1e-6


@pytest.mark.parametrize('level', [0, 2000, 65535])
def test_flat_and_saturated_images_do_not_produce_a_beam(level):
    assert not len(fit_single_beam(np.full((80, 100), level, np.uint16)).spots)


def test_selected_region_uses_sensor_offset_and_ignores_outside_pixels():
    viewer = Viewer.__new__(Viewer)
    viewer.camera = SimpleNamespace(ROI=(1200, 1000, 80, 40))
    viewer.fit_rect_sensor = (180, 140, 1204, 908)
    frame = np.full((1000, 1200), 65535, np.uint16)
    frame[100:868, 100:1124] = beam()
    spot = viewer._profile(frame)[0]
    assert spot['x'] == pytest.approx(600, abs=.5)
    assert spot['y'] == pytest.approx(490, abs=.5)
    viewer.fit_rect_sensor = (2000, 2000, 2100, 2100)
    assert not len(viewer._profile(frame))
    assert 'outside' in viewer.profiler_message


def test_p_toggles_profiler_independently_of_array_fitting(monkeypatch):
    viewer = Viewer.__new__(Viewer)
    viewer.profiler_enabled = False
    viewer.do_fitting = False
    viewer.latest_spots = SpotArray.empty()
    monkeypatch.setattr('beam_profiler.ui.viewer.write_status', lambda _: None)
    assert viewer._handle_key(ord('P'))
    assert viewer.profiler_enabled
    assert not viewer.do_fitting
    assert viewer._handle_key(ord('p'))
    assert not viewer.profiler_enabled


def test_asymmetric_core_and_halo_warn_and_are_stable_with_dark_padding():
    y, x = np.mgrid[:436, :588]
    image = (16 + 30000 * np.exp(-.5 * (((x-175)/7)**2 + ((y-266)/6)**2))
             + 9000 * np.exp(-.5 * (((x-151)/30)**2 + ((y-274)/25)**2))).astype(np.uint16)
    original = fit_single_beam(image)
    padded = fit_single_beam(np.pad(image, 100, constant_values=16))
    assert 'Poor Gaussian match' in original.message
    assert 'Poor Gaussian match' in padded.message
    # Adding dark margins must not change which part of the beam is fitted.
    assert padded.spots.x[0] - 100 == pytest.approx(original.spots.x[0], abs=.5)
    assert padded.spots.y[0] - 100 == pytest.approx(original.spots.y[0], abs=.5)
    assert padded.spots.sigma_0[0] == pytest.approx(original.spots.sigma_0[0], rel=.02)


def test_true_gaussian_is_not_flagged_as_poor_match():
    assert 'Poor' not in fit_single_beam(beam()).message


@pytest.mark.parametrize('angle', [0., .4, -1.2])
@pytest.mark.parametrize('sigmas', [(.15, .08), (.1, .1)])
def test_exact_gaussian_derivatives_match_central_differences(angle, sigmas):
    from beam_profiler.processing.profiler import _GaussianEvaluator
    y, x = np.mgrid[-.4:.4:17j, -.5:.5:23j]
    p = np.array([.03, 1.2, .1, -.05, *np.log(sigmas), angle])
    evaluator = _GaussianEvaluator(x.ravel(), y.ravel(), np.zeros(x.size))
    exact = evaluator.jacobian(p).copy()
    numerical = np.empty_like(exact)
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = 1e-6
        numerical[:, i] = (evaluator.residual(p+delta) - evaluator.residual(p-delta)) / 2e-6
    np.testing.assert_allclose(exact, numerical, rtol=2e-6, atol=2e-9)


def test_evaluator_cache_invalidates_when_parameter_array_is_mutated():
    from beam_profiler.processing.profiler import _GaussianEvaluator
    x = np.linspace(-.5, .5, 25)
    evaluator = _GaussianEvaluator(x, x, np.zeros(25))
    p = np.array([0., 1., .0, .0, -2., -3., .4])
    before = evaluator.residual(p).copy()
    jac_before = evaluator.jacobian(p).copy()
    p[2] += .1
    after = evaluator.residual(p)
    assert not np.allclose(before, after)
    assert not np.allclose(jac_before, evaluator.jacobian(p))


@pytest.mark.parametrize('cx', [160, 500])
def test_exact_derivatives_preserve_numerical_solver_result(cx, monkeypatch):
    from beam_profiler.processing import profiler
    frame = beam(cx=cx)
    exact = profiler.fit_single_beam(frame)
    solver = profiler.least_squares
    def numerical(*args, **kwargs):
        kwargs['jac'] = '2-point'
        return solver(*args, **kwargs)
    monkeypatch.setattr(profiler, 'least_squares', numerical)
    reference = profiler.fit_single_beam(frame)
    for field in ['x', 'y', 'sigma_0', 'sigma_1', 'I0']:
        np.testing.assert_allclose(getattr(exact.spots, field), getattr(reference.spots, field), rtol=1e-6, atol=1e-6)
    assert exact.message == reference.message
