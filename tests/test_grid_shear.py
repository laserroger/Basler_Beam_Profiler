"""Geometric ground truth for the H overlay and the shared HTTP measurement."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from beam_profiler.processing.shear import fit_grid_shear
from beam_profiler.processing.spots import SpotArray
from beam_profiler.processing.measurements import SpotStatistics
from beam_profiler.processing.pipeline import FrameProcessor
from beam_profiler.session import LiveSession, SessionSnapshot
from beam_profiler.server import create_app
from beam_profiler.ui import overlays, viewer


def grid(rotation=0., shear=0., missing=False):
    # Deliberately unequal spacing on both axes.
    x = np.array([0, 60, 127, 200, 280, 365, 457.])
    y = np.array([0, 72, 151, 233, 320, 413, 512.])
    row, col = np.indices((7, 7))
    row, col = row.ravel(), col.ravel()
    t, s = np.radians([rotation, shear])
    u = np.array([np.cos(t), np.sin(t)])
    v = np.array([-np.sin(t - s), np.cos(t - s)])
    xy = x[col, None] * u + y[row, None] * v + [800, 1100]
    if missing:
        keep = np.arange(49) != 17
        xy, row, col = xy[keep], row[keep], col[keep]
    n = len(xy)
    spots = SpotArray(xy[:, 0], xy[:, 1], np.ones(n), np.ones(n),
                      np.tile([1., 0.], (n, 1)), np.tile([0., 1.], (n, 1)),
                      np.ones(n), np.ones(n))
    return spots, [spots[row == i] for i in range(7)], [spots[col == i] for i in range(7)]


@pytest.mark.parametrize('rotation', [-24., 0., 31.])
def test_rotation_and_unequal_spacing_are_not_shear(rotation):
    _, rows, columns = grid(rotation=rotation)
    fit = fit_grid_shear(rows, columns)
    assert fit['valid']
    assert fit['shear_deg'] == pytest.approx(0., abs=1e-10)
    assert fit['rotation_deg'] == pytest.approx(rotation, abs=1e-10)
    assert fit['orthogonal_rms_px'] < 1e-5


@pytest.mark.parametrize('shear', [-.5, .5])
@pytest.mark.parametrize('rotation', [-20., 0., 25.])
@pytest.mark.parametrize('missing', [False, True])
def test_signed_shear_survives_rotation_missing_spot_and_unequal_spacing(shear, rotation, missing):
    _, rows, columns = grid(rotation, shear, missing)
    fit = fit_grid_shear(rows, columns)
    assert fit['valid']
    assert fit['spot_count'] == (48 if missing else 49)
    assert fit['shear_deg'] == pytest.approx(shear, abs=1e-9)
    assert fit['angle_deg'] == pytest.approx(90 - shear, abs=1e-9)
    assert fit['line_rms_px'] < 1e-5
    assert fit['orthogonal_rms_px'] > .5


def test_invalid_grids_do_not_report_zero_shear():
    _, rows, columns = grid()
    for rr, cc in [([], []), (rows[:1], columns), (rows[:-1], columns),
                   (rows + [rows[0]], columns + [columns[0]])]:
        result = fit_grid_shear(rr, cc)
        assert not result['valid']
        assert 'reason' in result
        assert 'shear_deg' not in result
    rows[0].x[0] = np.nan
    assert not fit_grid_shear(rows, columns)['valid']


def test_live_statistics_reset_and_saturation():
    spots, _, _ = grid(rotation=1., shear=.5)
    stats = SpotStatistics()
    stats.update(spots[np.random.default_rng(4).permutation(49)])
    assert stats.grid_shear['valid']
    assert stats.grid_shear['shear_deg'] == pytest.approx(.5, abs=1e-9)
    stats.update(spots, saturated=np.arange(49) == 5)
    assert not stats.grid_shear['valid']
    assert 'Saturated' in stats.grid_shear['reason']
    stats.update(SpotArray.empty())
    assert not stats.grid_shear['valid']


def test_api_exposes_the_frame_shear_without_h_enabled():
    spots, _, _ = grid(rotation=.5, shear=.49)
    stats = SpotStatistics()
    stats.update(spots)
    frame = np.zeros((20, 20), np.uint8)
    result = FrameProcessor().process(frame, roi=(20, 20, 0, 0), pixel_size=1e-6, full_scale=255)
    result = replace(result, spots=spots, statistics=stats, saturated=np.zeros(49, dtype=bool))
    session = LiveSession()
    session.publish(SessionSnapshot(result=result))
    payload = create_app(session).test_client().get('/api/spots').json
    assert payload['stats']['grid_shear']['shear_deg'] == pytest.approx(.49, abs=1e-9)
    assert payload['timestamp'] == result.timestamp
    session.publish(SessionSnapshot(result=replace(result, statistics=SpotStatistics())))
    assert not create_app(session).test_client().get('/api/spots').json['stats']['grid_shear']['valid']
    assert result.statistics.grid_shear['valid']


def test_h_toggles_shear_with_grid_even_when_g_is_off(monkeypatch):
    spots, _, _ = grid(rotation=.5, shear=.49)
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.statistics = SpotStatistics()
    v.statistics.update(spots)
    v.camera = SimpleNamespace(ROI=(1600, 1700, 0, 0), pixel_size=1e-6)
    v.analysis = SimpleNamespace(saturated=np.zeros(49, dtype=bool))
    v.profiler_enabled = v.row_col_fitting = v.show_stats = v.show_rect_stats = False
    v.show_displacement = False
    v.do_fitting = True
    v._hud_lines = lambda: []
    v._draw_rectangles = lambda *args: None
    monkeypatch.setattr(viewer, 'write_status', lambda _: None)
    seen = []
    monkeypatch.setattr(overlays, 'draw_grid_shear', lambda img, fit, y: seen.append(fit) or y)
    for name in ('draw_spots', 'draw_row_col'):
        monkeypatch.setattr(overlays, name, lambda *args, **kwargs: None)
    frame = np.zeros((1700, 1600), np.uint8)
    v._compose(frame, spots)
    assert not seen
    v._handle_key(ord('h'))
    v._compose(frame, spots)
    assert len(seen) == 1 and seen[0]['valid']
    assert not v.show_stats
    v._handle_key(ord('H'))
    v._compose(frame, spots)
    assert len(seen) == 1


def test_overlay_line_uses_fitted_direction():
    _, rows, columns = grid(rotation=12, shear=.5)
    fit = fit_grid_shear(rows, columns)
    for group, direction, axis in [(rows[0], fit['row_direction'], 0),
                                    (columns[0], fit['column_direction'], 1)]:
        a, b = overlays._grid_line_endpoints(group, direction, axis)
        delta = b - a
        np.testing.assert_allclose(delta / np.linalg.norm(delta), direction, atol=1e-12)


def test_displacement_zero_for_rotated_sheared_uneven_grid():
    from beam_profiler.processing.displacement import grid_displacement
    _, rows, columns = grid(rotation=12., shear=.5, missing=True)
    result = grid_displacement(rows, columns, fit_grid_shear(rows, columns))
    assert result['valid']
    assert result['rms_px'] < 1e-10
    np.testing.assert_allclose(result['residual'], 0., atol=1e-10)


def test_displacement_preserves_known_bow_and_reconstructs_spots():
    from beam_profiler.processing.displacement import grid_displacement
    spots, _, _ = grid(rotation=0.)
    # Symmetric y positions make the common quadratic bow orthogonal to tilt.
    y = np.repeat(np.linspace(-210, 210, 7), 7)
    spots.y = 1200 + y
    bow = .03 * ((y/210)**2 - np.mean((y/210)**2))
    spots.x += bow
    rows = [spots[i*7:(i+1)*7] for i in range(7)]
    columns = [spots[np.arange(49) % 7 == i] for i in range(7)]
    result = grid_displacement(rows, columns, fit_grid_shear(rows, columns))
    xy = np.column_stack((spots.x, spots.y));order = np.lexsort((xy[:,1], xy[:,0]))
    np.testing.assert_allclose(result['reference']+result['residual'],xy[order],atol=1e-10)
    np.testing.assert_allclose(result['residual'][:,0],bow[order],atol=1e-10)
    np.testing.assert_allclose(result['residual'][:,1],0.,atol=1e-10)
    assert not grid_displacement([],[],{'valid':False,'reason':'no grid'})['valid']


def test_e_is_independent_and_h_does_not_hide_beams(monkeypatch):
    spots, _, _ = grid(rotation=.5, shear=.49)
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.statistics = SpotStatistics();v.statistics.update(spots)
    v.camera = SimpleNamespace(ROI=(1600,1700,0,0),pixel_size=1e-6)
    v.analysis = SimpleNamespace(saturated=np.zeros(49,dtype=bool))
    v.profiler_enabled = v.show_stats = v.show_rect_stats = v.show_displacement = False
    v.do_fitting = v.row_col_fitting = True
    v.fit_rect_sensor = None
    v._hud_lines=lambda:[];v._draw_rectangles=lambda *args:None
    monkeypatch.setattr(viewer,'write_status',lambda _:None)
    calls=[]
    for name in ('draw_spots','draw_row_col','draw_displacement'):
        monkeypatch.setattr(overlays,name,lambda *args,_name=name,**kwargs:calls.append((_name,kwargs)))
    frame=np.zeros((1700,1600),np.uint8)
    v._compose(frame,spots)
    assert [x[0] for x in calls]==['draw_row_col','draw_spots']
    assert calls[-1][1]['show_centers']
    v._handle_key(ord('e'));calls.clear();v._compose(frame,spots)
    assert [x[0] for x in calls]==['draw_row_col','draw_displacement','draw_spots']
    assert v.do_fitting and v.row_col_fitting and not v.show_stats
    v._handle_key(ord('H'));calls.clear();v._compose(frame,spots)
    assert [x[0] for x in calls]==['draw_displacement','draw_spots']
    v._handle_key(ord('E'));calls.clear();v._compose(frame,spots)
    assert [x[0] for x in calls]==['draw_spots']


def test_displacement_arrows_are_exactly_1000x(monkeypatch):
    from beam_profiler.roi import ViewTransform
    drawn=[]
    monkeypatch.setattr(overlays.cv2,'arrowedLine',lambda image,a,b,*args,**kw:drawn.append((a,b)))
    result={'valid':True,'reference':np.array([[50.,60.]]),'residual':np.array([[.03,-.02]]),
            'rows':np.array([0]),'columns':np.array([0])}
    overlays.draw_displacement(np.zeros((200,200,3),np.uint8),ViewTransform((200,200,100,200),200),result)
    assert drawn==[((50,60),(80,40))]
