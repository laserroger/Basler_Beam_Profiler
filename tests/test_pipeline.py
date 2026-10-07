"""Exercise the actual pipeline without camera drivers or a GUI."""
import numpy as np
import pytest
from beam_profiler.processing.pipeline import FrameProcessor, AnalysisOptions
from beam_profiler.synthetic import render as render_spots, Spot


def test_old_results_keep_their_own_measurements():
    p = FrameProcessor()
    frame = render_spots(300,200,[Spot(90,100,8,8),Spot(190,100,8,8)],noise=0)
    result = p.process(frame,roi=(300,200,20,40),pixel_size=3.45e-6,full_scale=65535,
                       options=AnalysisOptions(fitting=True))
    assert len(result.spots) == 2
    before = result.statistics.summary(result.pixel_size)
    second = p.process(np.zeros_like(frame),roi=(300,200,20,40),pixel_size=3.45e-6,full_scale=65535)
    assert len(second.spots) == 0
    assert len(result.spots) == 2
    assert result.statistics.summary(result.pixel_size) == before
    assert result.statistics is not second.statistics


def test_region_offsets_saturation_and_stats_share_one_frame():
    frame = render_spots(300,200,[Spot(100,100,15,10,amplitude=1.5)],noise=0)
    options = AnalysisOptions(profiler=True,fit_rect=(550,630,670,770),
                              rect=(550,630,670,770),rect_stats=True)
    result = FrameProcessor().process(frame,roi=(300,200,500,600),pixel_size=3.45e-6,
                                      full_scale=65535,options=options)
    assert result.spots.x[0] == pytest.approx(100,abs=1)
    assert result.saturated.tolist() == [True]
    assert result.rect_stats['max'] == 65535
    assert result.options.fit_rect == options.fit_rect


def test_mismatched_roi_is_rejected():
    with pytest.raises(ValueError,match='ROI'):
        FrameProcessor().process(np.zeros((10,10),np.uint8),roi=(20,10,0,0),pixel_size=1,full_scale=255)
