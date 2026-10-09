"""Disconnect paths must stop stale publication and allow orderly shutdown."""
from types import SimpleNamespace
import numpy as np
import pytest
from beam_profiler.ui import viewer
from beam_profiler.session import LiveSession, SessionSnapshot
from beam_profiler.processing.spots import SpotArray


def make_viewer(monkeypatch, grab, exposure=1000):
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.camera = SimpleNamespace(ExposureTime=exposure, grab_image=grab)
    v._quit_requested = False
    v.session = LiveSession()
    v.session.publish(SessionSnapshot(status={'camera_model':'test'}))
    v.current_frame = np.ones((10,10),np.uint16)
    v.latest_spots = SpotArray.empty()
    v._native_key_monitor = None
    monkeypatch.setattr(viewer.cv2,'imshow',lambda *args: None)
    return v


def test_disconnect_exception_requests_exit_and_clears_stale_results(monkeypatch):
    def fail(): raise RuntimeError('USB camera removed')
    v = make_viewer(monkeypatch,fail)
    assert v._grab_frame() is None
    assert v._quit_requested
    assert v.current_frame is None
    assert v.session.read().result is None
    assert 'USB camera removed' in v.session.read().status['error']


def test_missing_frame_requests_immediate_exit(monkeypatch):
    v = make_viewer(monkeypatch,lambda: None)
    assert v._grab_frame() is None
    assert v._quit_requested
    assert v.current_frame is None
    assert v.session.read().result is None


def test_good_frame_keeps_running(monkeypatch):
    frame = np.zeros((10,10),np.uint16)
    v = make_viewer(monkeypatch,lambda: frame)
    assert v._grab_frame() is frame
    assert not v._quit_requested


def test_loop_exits_immediately_without_retrying_missing_frames(monkeypatch,tmp_path):
    calls = []
    def grab():
        calls.append(1)
        return None
    v = make_viewer(monkeypatch,grab)
    monkeypatch.setattr(viewer,'DATA_DIR',str(tmp_path))
    monkeypatch.setattr(viewer.settings,'pump_events',lambda: None)
    monkeypatch.setattr(viewer.cv2,'getWindowProperty',lambda *args: 1)
    v._apply_web_commands = v._apply_scroll = lambda: None
    v._run_loop()
    assert calls == [1]
    assert v._quit_requested


def test_gui_cleanup_continues_after_opencv_failure(monkeypatch):
    v = make_viewer(monkeypatch,lambda: None)
    cleaned = []
    v._run_loop = lambda **kwargs: None
    def fail(): raise RuntimeError('window already gone')
    monkeypatch.setattr(viewer.cv2,'destroyAllWindows',fail)
    monkeypatch.setattr(viewer.settings,'destroy_settings',lambda: cleaned.append('tk'))
    with pytest.raises(RuntimeError,match='window already gone'):
        v.run()
    assert cleaned == ['tk']
