from types import SimpleNamespace

import cv2
import pytest

from beam_profiler.roi import ROIModel
from beam_profiler.ui.viewer import Viewer


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setattr('beam_profiler.ui.viewer.sys.platform', 'darwin')
    v = Viewer.__new__(Viewer)
    v.camera = SimpleNamespace(ROI=(2048, 1536, 0, 0))
    v.roi_model = ROIModel(2048, 1536)
    v._mouse_display = (650, 550)
    v._pending_scroll = [0., 0.]
    v.current_frame = None
    return v


def test_cocoa_scroll_uses_delta_and_preserves_pointer_anchor(viewer):
    before = viewer.view.to_sensor(*viewer._mouse_display)
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 20, 0, None)
    assert viewer._mouse_display == (650, 550)
    assert viewer.camera.ROI[0] == 2048  # queued until next frame
    viewer._apply_scroll()
    assert viewer.camera.ROI[0] < 2048
    assert viewer.view.to_sensor(*viewer._mouse_display) == pytest.approx(before, abs=2)
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, -20, 0, None)
    viewer._apply_scroll()
    assert viewer.camera.ROI[0] >= 2045


def test_shift_changes_aspect_in_both_directions(viewer):
    original = viewer.roi_model.aspect
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 20, cv2.EVENT_FLAG_SHIFTKEY, None)
    viewer._apply_scroll()
    assert viewer.roi_model.aspect < original
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, -40, cv2.EVENT_FLAG_SHIFTKEY, None)
    viewer._apply_scroll()
    assert viewer.roi_model.aspect > original


def test_zero_delta_and_padding_do_not_zoom(viewer):
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 0, cv2.EVENT_FLAG_SHIFTKEY, None)
    viewer._apply_scroll()
    assert viewer.camera.ROI == (2048, 1536, 0, 0)
    viewer._mouse_display = (50, 0)
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 20, 0, None)
    viewer._apply_scroll()
    assert viewer.camera.ROI == (2048, 1536, 0, 0)


def test_windows_signed_wheel_delta_not_modifier(viewer, monkeypatch):
    monkeypatch.setattr('beam_profiler.ui.viewer.sys.platform', 'win32')
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 650, 550, 120 << 16, None)
    viewer._apply_scroll()
    smaller = viewer.camera.ROI[0]
    viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 650, 550, ((-120 & 0xffff) << 16), None)
    viewer._apply_scroll()
    assert viewer.camera.ROI[0] > smaller


def test_trackpad_burst_is_combined_and_bounded(viewer):
    for _ in range(100):
        viewer.on_mouse(cv2.EVENT_MOUSEWHEEL, 0, 20, 0, None)
    viewer._apply_scroll()
    assert 1400 < viewer.camera.ROI[0] < 1500
    roi = viewer.camera.ROI
    viewer._apply_scroll()
    assert viewer.camera.ROI == roi
