import cv2
import numpy as np
import pytest

from beam_profiler.roi import ViewTransform
from beam_profiler.ui.viewer import Viewer
from types import SimpleNamespace


@pytest.mark.parametrize('shape,window', [((72, 80), 1100), ((71, 83), 1100), ((200, 320), 110)])
def test_overlay_matches_resized_impulse_centroid(shape, window):
    h, w = shape
    x, y = w//2, h//2
    # A symmetric, well-sampled spot supplies an independent image-space check.
    yy, xx = np.mgrid[:h, :w]
    raw = np.exp(-((xx-x)**2+(yy-y)**2)/(2*3.0**2)).astype(np.float32)
    v = ViewTransform((w, h, 100, 200), window)
    resized = cv2.resize(raw, (v.view_w, v.view_h), interpolation=cv2.INTER_LINEAR)
    ry, rx = np.mgrid[:v.view_h, :v.view_w]
    measured = ((rx*resized).sum()/resized.sum()+v.pad_l,
                (ry*resized).sum()/resized.sum()+v.pad_t)
    expected = v.roi_to_display(x, y)
    np.testing.assert_allclose(expected, measured, atol=.6)
    bx, by = v.roi_to_display_many([x], [y])
    assert (bx[0], by[0]) == expected
    np.testing.assert_allclose(v.to_sensor(*expected), (100+x, 200+y), atol=.51/min(v.scale_x, v.scale_y))


def test_hover_at_zoomed_pixel_edges_and_center():
    v = Viewer.__new__(Viewer)
    v.camera = SimpleNamespace(ROI=(80, 72, 968, 816))
    v.current_frame = np.arange(72*80, dtype=np.uint16).reshape(72, 80)
    transform = v.view
    for x, y in [(0, 0), (35, 43), (79, 71)]:
        cx, cy = transform.roi_to_display(x, y)
        for delta in [-4, 0, 4]:
            v._mouse_display = (cx+delta, cy+delta)
            assert v._hover_pixel() == (968+x, 816+y, int(v.current_frame[y, x]))
