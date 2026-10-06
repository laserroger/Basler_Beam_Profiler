import numpy as np
import pytest
from beam_profiler.processing.blobs import candidate_mask, detect_spots, DETECT_MAX_PIXELS
from beam_profiler.synthetic import Spot, render


@pytest.mark.parametrize('shape', [(100, 5320), (5320, 100), (1024, 1024)])
def test_roi_within_budget_keeps_full_resolution(shape):
    image = np.zeros(shape, np.uint16)
    image[20:40, 20:40] = 60000
    mask, scale = candidate_mask(image)
    assert scale == 1.
    assert mask.shape == shape


@pytest.mark.parametrize('shape', [(1200, 1600), (600, 5320), (5320, 600)])
def test_larger_roi_uses_total_pixel_budget(shape):
    image = np.zeros(shape, np.uint16)
    image[20:60, 20:60] = 60000
    mask, scale = candidate_mask(image)
    assert scale == pytest.approx(np.sqrt(DETECT_MAX_PIXELS / image.size))
    assert mask.size <= DETECT_MAX_PIXELS
    assert mask.shape == tuple(int(d * scale) for d in shape)


@pytest.mark.parametrize('transpose', [False, True])
def test_small_spots_survive_in_long_strip(transpose):
    truth = [Spot(float(x), 50., sigma_x=2.5, amplitude=.45 + .3*(i%2))
             for i, x in enumerate(np.linspace(100, 5200, 24))]
    frame = render(5320, 100, truth, rng=np.random.default_rng(7))
    if transpose:
        frame = frame.T.copy()
    found = detect_spots(frame)
    assert len(found) == len(truth)
    for s in truth:
        x, y = (s.y, s.x) if transpose else (s.x, s.y)
        assert np.min(np.hypot(found.x-x, found.y-y)) < .2
