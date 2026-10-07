import cv2
import numpy as np
from beam_profiler import fitconfig
from beam_profiler.processing.pipeline import FrameProcessor, AnalysisOptions
from beam_profiler.session import LiveSession, SessionSnapshot
from beam_profiler.server import create_app
from beam_profiler.ui.viewer import Viewer


def snapshot(value=32768, roi=(40,20,100,200)):
    frame = np.full((roi[1],roi[0]),value,np.uint16)
    options = AnalysisOptions(rect=(105,205,115,215))
    result = FrameProcessor().process(frame,roi=roi,pixel_size=3.45e-6,full_scale=65535,options=options)
    return SessionSnapshot(result=result,status={'roi':roi})


def test_api_needs_no_viewer_and_returns_frame_measurements():
    session = LiveSession()
    snap = snapshot()
    session.publish(snap)
    client = create_app(session).test_client()
    data = client.get('/api/rect_stats').json
    assert data['pixel_stats']['mean'] == 32768
    assert data['timestamp'] == snap.result.timestamp
    assert client.get('/api/spots').json['timestamp'] == snap.result.timestamp
    assert client.get('/api/status').json['roi'] == [40,20,100,200]
    image = cv2.imdecode(np.frombuffer(client.get('/api/image_within_rect').data,np.uint8),0)
    assert image.shape == (10,10)
    assert image.mean() == 128


def test_read_pins_snapshot_despite_next_frame_publication():
    old, new = snapshot(), snapshot(60000,roi=(80,40,500,600))
    class AdvancingSession(LiveSession):
        reads = 0
        def read(self):
            self.reads += 1
            result = super().read()
            self.publish(new)
            return result
    session = AdvancingSession()
    session.publish(old)
    data = create_app(session).test_client().get('/api/rect_stats').json
    assert session.reads == 1
    assert data['pixel_stats']['mean'] == 32768
    assert data['timestamp'] == old.result.timestamp


def test_controls_apply_on_acquisition_thread_not_http_thread():
    session = LiveSession()
    session.publish(snapshot())
    client = create_app(session).test_client()
    assert client.post('/api/set_fit_rect',json={'coords':[1,2,30,40]}).status_code == 200
    original = fitconfig.active().heatmap
    assert client.put('/api/fit_config',json={'heatmap':not original}).status_code == 200
    assert fitconfig.active().heatmap == original
    v = Viewer.__new__(Viewer)
    v.session = session
    v.fit_rect_sensor = None
    v._apply_web_commands()
    assert v.fit_rect_sensor == (1,2,30,40)
    assert fitconfig.active().heatmap != original
    assert session.take_commands() == []
    # The published snapshot still describes the captured frame, not commands.
    assert session.read().result.options.fit_rect is None


def test_invalid_controls_and_no_frame():
    session = LiveSession()
    client = create_app(session).test_client()
    assert client.get('/api/image').status_code == 400
    assert client.get('/api/spots').json['spot_count'] == 0
    for coords in (5,[1,2],['bad',0,1,2]):
        assert client.post('/api/set_rect',json={'coords':coords}).status_code == 400
    assert client.put('/api/fit_config',json={'unknown':1}).status_code == 400
    assert session.take_commands() == []


def test_real_viewer_loop_publishes_the_result_it_draws(monkeypatch, tmp_path):
    from beam_profiler.ui import viewer
    from beam_profiler.cameras.simulated import SimulatedCamera
    monkeypatch.setattr(viewer, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(viewer.gpu, 'warm_up', lambda: None)
    monkeypatch.setattr(viewer.settings, 'prepare_gui', lambda: None)
    monkeypatch.setattr(viewer.settings, 'pump_events', lambda: None)
    if viewer.sys.platform == 'darwin':
        from beam_profiler.ui import macos
        monkeypatch.setattr(macos, 'attach_close_button', lambda *args: None)
        monkeypatch.setattr(macos, 'attach_key_monitor', lambda *args: None)
    for name in ('namedWindow', 'resizeWindow', 'setMouseCallback', 'imshow'):
        monkeypatch.setattr(viewer.cv2, name, lambda *args: None)
    monkeypatch.setattr(viewer.cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(Viewer, '_poll_keys', lambda *args: True)
    monkeypatch.setattr(Viewer, '_sleep_for_fps', lambda *args: None)
    v = Viewer([SimulatedCamera(width=320,height=240,grid=(2,1),noise=0)])
    v.do_fitting = True
    v.session.set_region('rect_sensor',(20,20,60,60))
    v._run_loop(max_frames=2)
    result = v.session.read().result
    assert result is v.analysis
    assert result.spots is v.latest_spots
    assert len(result.spots) == 2
    assert result.options.rect == (20,20,60,60)
    assert create_app(v.session).test_client().get('/api/spots').json['spot_count'] == 2
