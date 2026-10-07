"""Saturation warnings use raw data and preserve per-beam attribution."""
import cv2
import numpy as np
import pytest

from beam_profiler.processing.saturation import saturated_spots
from beam_profiler.processing.spots import SpotArray
from beam_profiler.processing.profiler import fit_single_beam
from beam_profiler.roi import ViewTransform
from beam_profiler.ui.overlays import draw_spots


def spots():
    return SpotArray([25, 75], [50, 50], [12, 12], [8, 8],
                     [[1, 0], [1, 0]], [[0, 1], [0, 1]], [100, 100], [50, 50])


@pytest.mark.parametrize('dtype,maximum,clipped', [
    (np.uint8,255,255), (np.uint16,65535,65535),
    (np.uint16,65535,65520), (np.uint16,65535,65472)])
def test_raw_threshold_per_spot(dtype, maximum, clipped):
    frame = np.zeros((100,100),dtype=dtype)
    frame[50,25] = clipped
    frame[50,75] = int(maximum * .998)
    assert saturated_spots(frame,spots(),maximum).tolist() == [True,False]
    # Saturation elsewhere in the frame doesn't falsely flag either spot.
    frame[50,25] = 0
    frame[5,5] = clipped
    assert not saturated_spots(frame,spots(),maximum).any()


def test_ellipse_excludes_bounding_box_corner():
    frame = np.zeros((100,100),np.uint8)
    frame[57,36] = 255
    assert not saturated_spots(frame,spots(),255).any()


def test_warning_is_red_and_thicker_than_green():
    image = np.zeros((100,100,3),np.uint8)
    draw_spots(image,spots(),ViewTransform((100,100,500,600),100),1e-6,
               min_label_radius=100,saturated=[True,False])
    red = np.all(image[:,:50] == (0,0,255),axis=2).sum()
    green = np.all(image[:,50:] == (0,255,0),axis=2).sum()
    assert red > green * 1.4
    assert not np.any(np.all(image[:,:50] == (0,255,0),axis=2))


def test_saturated_gaussian_retains_warning_geometry():
    y,x = np.mgrid[:200,:240]
    frame = np.clip(1000 + 90000*np.exp(-((x-120)**2/25**2+(y-100)**2/18**2)/2),0,65535).astype(np.uint16)
    result = fit_single_beam(frame)
    assert len(result.spots) == 1
    assert saturated_spots(frame,result.spots,65535).tolist() == [True]
    assert 'Saturated' in result.message
