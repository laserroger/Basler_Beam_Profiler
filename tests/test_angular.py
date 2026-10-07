import numpy as np
import pytest
from beam_profiler.angular import response_from_centers, AngularMonitor
from beam_profiler.server import create_app
from beam_profiler.session import LiveSession, SessionSnapshot
from beam_profiler.processing.pipeline import FrameProcessor, AnalysisOptions


def test_index_spacing_accepts_any_count():
    for n in (2, 7, 61):
        x = 100 + 50*np.arange(n) + .1*np.arange(n)**2
        d = response_from_centers(x[::-1])
        np.testing.assert_allclose(d['pair_index'], np.arange(1,n))
        np.testing.assert_allclose(d['response'], np.diff(x)[::-1]*3.45/150)
        assert len(d['response']) == n-1
        assert 'frequency_mhz' not in d
        assert d['residual_rms'] < 1e-12


def test_invalid_centers_and_incomplete_roi():
    with pytest.raises(ValueError):
        response_from_centers([1,1,2])
    m = AngularMonitor()
    m.update(np.zeros((8,20),np.uint16), (20,8,0,0), (0,0,30,8), None, 1)
    assert not m.latest['valid']
    assert 'entire green' in m.latest['error']


def test_api_and_page():
    monitor = AngularMonitor()
    session = LiveSession()
    session.publish(SessionSnapshot(angular=monitor.latest))
    client = create_app(session).test_client()
    assert client.get('/angular').status_code == 200
    assert client.get('/api/angular').json['valid'] is False
    client = create_app(LiveSession()).test_client()
    assert client.get('/api/angular').status_code == 409


def test_combined_page_and_camera_preview():
    import cv2
    frame = np.full((20,40),32768,dtype=np.uint16)
    session = LiveSession()
    result = FrameProcessor().process(frame, roi=(40,20,100,200), pixel_size=3.45e-6,
                                      full_scale=65535,
                                      options=AnalysisOptions(fit_rect=(105,205,130,210)))
    session.publish(SessionSnapshot(result=result, angular=AngularMonitor().latest))
    client = create_app(session).test_client()
    page = client.get('/').text
    assert 'id="camera"' in page and 'id="ymin"' in page and 'id="ymax"' in page
    response = client.get('/api/preview')
    image = cv2.imdecode(np.frombuffer(response.data,np.uint8),cv2.IMREAD_COLOR)
    assert image.shape == (20,40,3)
    assert 'no-store' in response.headers['Cache-Control']
    assert image[5,5,1] > image[5,5,0]
    np.testing.assert_array_equal(frame,32768)


def test_clearing_green_box_removes_previous_curve_without_fitting(monkeypatch):
    import beam_profiler.angular as angular
    monitor = AngularMonitor()
    monitor.latest = dict(valid=True, response=[1,2])
    def forbidden(*args, **kwargs):
        raise AssertionError('No detection may run without a green rectangle')
    monkeypatch.setattr(angular, 'detect_spots', forbidden)
    monitor.update(np.ones((8,20),np.uint16), (20,8,0,0), None, None, 1.)
    assert monitor.latest['valid'] is False
    assert 'response' not in monitor.latest
    assert 'green rectangle' in monitor.latest['error']


def test_typed_y_limits():
    from beam_profiler.ui.spacing import parse_limits
    assert parse_limits('1', '1.4') == (1.,1.4)
    assert parse_limits('0.9','1.5') == (.9,1.5)
    for low,high in [('a','1.4'),('1.4','1'),('1','1'),('nan','1.4'),('1','inf')]:
        with pytest.raises(ValueError):
            parse_limits(low,high)


def test_monitor_uses_focal_setting_and_camera_pixel_pitch(monkeypatch):
    import beam_profiler.angular as angular
    from beam_profiler.fitconfig import FitConfig
    monkeypatch.setattr(angular, 'detect_spots',
                        lambda *args, **kw: np.array([(2.,), (12.,)], dtype=[('x', float)]).view(np.recarray))
    monitor = AngularMonitor()
    frame = np.ones((8, 20), np.uint16)
    for focal in (150., 300.):
        monitor.update(frame, (20, 8, 0, 0), (0, 0, 20, 8),
                       FitConfig(angular_focal_mm=focal), 1., pixel_um=5.)
        assert monitor.latest['valid']
        assert monitor.latest['response'] == pytest.approx([50./focal])
        assert monitor.latest['pixel_um'] == 5.
        assert monitor.latest['focal_mm'] == focal
