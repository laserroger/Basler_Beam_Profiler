"""Headless frame analysis shared by the desktop, simulator and HTTP consumers.

The processor owns smoothing history. Each call returns a complete result whose
arrays and statistics belong to that frame; consumers must treat them as read-only.
Acquisition and display remain outside this module.
"""
from copy import copy
from dataclasses import dataclass
import time

import numpy as np

from .. import fitconfig
from .blobs import detect_spots
from .spots import SpotArray
from .profiler import fit_single_beam, BeamFit
from .stats import crop_rect, region_stats
from .saturation import saturated_spots
from .measurements import SpotStatistics


@dataclass(frozen=True)
class AnalysisOptions:
    fitting: bool = False
    profiler: bool = False
    fit_rect: tuple | None = None
    rect: tuple | None = None
    rect_stats: bool = False


@dataclass(frozen=True)
class FrameResult:
    frame: np.ndarray
    roi: tuple
    pixel_size: float
    options: AnalysisOptions
    spots: SpotArray
    statistics: SpotStatistics
    saturated: np.ndarray
    rect_stats: dict | None
    profiler_message: str
    timestamp: float


def profile_region(frame, roi, fit_rect):
    x0 = y0 = 0
    if fit_rect is not None:
        cropped = crop_rect(frame, roi, fit_rect)
        if cropped is None:
            return BeamFit(SpotArray.empty(), 'Fitting region is outside the camera ROI')
        frame, (x0, y0, _, _) = cropped
    result = fit_single_beam(frame)
    result.spots.x += x0
    result.spots.y += y0
    return result


class FrameProcessor:
    def __init__(self):
        self.statistics = SpotStatistics()

    def process(self, frame, *, roi, pixel_size, full_scale, options=None, config=None):
        options = options or AnalysisOptions()
        config = config or fitconfig.active()
        roi = tuple(roi)
        if frame.shape != (roi[1], roi[0]):
            raise ValueError('Frame dimensions do not match the camera ROI')
        timestamp = time.time()
        message = ''
        if options.profiler:
            fitted = profile_region(frame, roi, options.fit_rect)
            spots, message = fitted.spots, fitted.message
        elif options.fitting:
            spots = detect_spots(frame, cfg=config)
            if options.fit_rect is not None and len(spots):
                x1, y1, x2, y2 = options.fit_rect
                sx, sy = roi[2] + spots.x, roi[3] + spots.y
                spots = spots[(sx >= x1) & (sx <= x2) & (sy >= y1) & (sy <= y2)]
        else:
            spots = SpotArray.empty()
        # update assigns new arrays; copy the state so earlier results never
        # acquire the next frame's smoothing values or geometry.
        statistics = copy(self.statistics)
        statistics.update(spots)
        self.statistics = statistics
        return FrameResult(
            frame, roi, pixel_size, options, spots, statistics,
            saturated_spots(frame, spots, full_scale),
            region_stats(frame, roi, options.rect, pixel_size) if options.rect_stats else None,
            message, timestamp,
        )
