import numpy as np
import pytest
from beam_profiler.processing.spots import SpotArray
from beam_profiler.ui.viewer import Viewer


def make_grid():
    row, col = np.indices((3, 6))
    x = (70 * col + .15 * row).ravel()
    y = (60 * row + .1 * col).ravel()
    n = len(x)
    return SpotArray(x, y, np.ones(n), np.ones(n),
                     np.tile([1., 0.], (n, 1)), np.tile([0., 1.], (n, 1)),
                     np.ones(n), np.ones(n))


def viewer():
    v = Viewer.__new__(Viewer)
    v.row_col_fitting = True
    v.profiler_enabled = False
    v.ema = .8
    v.std_dx_ema = v.std_dy_ema = v.sigma_std_ema = 0.
    return v


@pytest.mark.parametrize("show_grid", [False, True])
def test_grid_statistics_only_measure_adjacent_pairs_within_each_axis(show_grid):
    v = viewer()
    v.row_col_fitting = show_grid
    v._update_spot_stats(make_grid())
    assert len(v.rows) == 3 and len(v.columns) == 6
    assert len(v.stats_dx) == 15 and len(v.stats_dy) == 12
    np.testing.assert_allclose(v.stats_dx, 70.)
    np.testing.assert_allclose(v.stats_dy, 60.)
    assert v.std_dx_ema < 1e-10 and v.std_dy_ema < 1e-10


def test_detecting_grid_discards_non_grid_smoothing_history():
    v = viewer()
    v.row_col_fitting = False
    line = make_grid()[:6]
    line.y = np.array([0., 10., 0., 10., 0., 10.])
    v._update_spot_stats(line)
    assert not v._stats_grid_mode
    assert v.std_dy_ema > 5
    v._update_spot_stats(make_grid())
    assert v._stats_grid_mode
    assert v.std_dx_ema < 1e-10 and v.std_dy_ema < 1e-10


def test_h_only_changes_display_not_measurements():
    v = viewer()
    v.row_col_fitting = False
    spots = make_grid()
    spots.x += np.tile(np.arange(6)**2 * .1, 3)
    v._update_spot_stats(spots)
    before = (v.stats_dx.copy(), v.stats_dy.copy(), v.std_dx_ema, v.std_dy_ema)
    v.row_col_fitting = True
    v._update_spot_stats(spots)
    for old, new in zip(before, (v.stats_dx, v.stats_dy, v.std_dx_ema, v.std_dy_ema)):
        np.testing.assert_allclose(new, old)


def test_grid_reports_real_spacing_variation_and_ignores_input_order():
    v = viewer()
    spots = make_grid()
    spots.x += np.tile(np.arange(6)**2 * .1, 3)
    spots = spots[np.random.default_rng(12).permutation(len(spots))]
    v._update_spot_stats(spots)
    expected = np.tile(70 + np.diff(np.arange(6)**2 * .1), 3)
    np.testing.assert_allclose(np.sort(v.stats_dx), np.sort(expected))
    np.testing.assert_allclose(v.std_dx_ema, np.std(expected))
