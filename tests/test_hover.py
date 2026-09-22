from types import SimpleNamespace

import numpy as np

from beam_profiler.ui.viewer import Viewer


def test_hover_reads_raw_16bit_value_with_roi_offset():
    viewer = Viewer.__new__(Viewer)
    viewer.camera = SimpleNamespace(ROI=(100, 60, 400, 200))
    viewer.current_frame = np.zeros((60, 100), np.uint16)
    viewer.current_frame[30, 50] = 32176
    viewer._mouse_display = viewer.view.to_display(450, 230)
    assert viewer._hover_pixel() == (450, 230, 32176)
    viewer.current_frame[30, 50] = 12345
    assert viewer._hover_pixel()[2] == 12345


def test_hover_has_no_value_in_padding_or_before_first_frame():
    viewer = Viewer.__new__(Viewer)
    viewer.camera = SimpleNamespace(ROI=(100, 60, 0, 0))
    viewer.current_frame = np.zeros((60, 100), np.uint8)
    viewer._mouse_display = (20, 0)
    assert viewer._hover_pixel() is None
    viewer._mouse_display = None
    assert viewer._hover_pixel() is None
    viewer.current_frame = None
    assert viewer._hover_pixel() is None
