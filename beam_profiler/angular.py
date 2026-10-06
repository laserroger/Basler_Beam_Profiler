"""Single-frame angular dispersion; no temporal averaging or physical-model fit."""
import time
import numpy as np
from .processing import detect_spots
from .processing.stats import crop_rect


def response_from_centers(x, pixel_um=3.45, focal_mm=150.):
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError('Need at least two distinct spot centers inside the green box')
    if not np.isfinite([pixel_um, focal_mm]).all() or pixel_um <= 0 or focal_mm <= 0:
        raise ValueError('Pixel size and focal length must be positive')
    index = np.arange(1, len(x)+1)
    x = x[::-1]  # Pair 1 starts at the rightmost detected spot.
    mid = np.arange(1, len(x))
    y = np.abs(np.diff(x)) * pixel_um / focal_mm
    slope, intercept = np.polyfit(mid, y, 1) if len(y)>1 else (0., y[0])
    trend = slope * mid + intercept
    return dict(spot_index=index.tolist(), x_px=x.tolist(), pair_index=mid.tolist(),
                response=y.tolist(), trend=trend.tolist(), slope=float(slope),
                intercept=float(intercept), residual_rms=float(np.sqrt(np.mean((y-trend)**2))))


class AngularMonitor:
    def __init__(self):
        self.latest = {'valid': False, 'error': 'Waiting for first frame'}
        self.sequence = 0
        self.previous_time = None

    def update(self, frame, roi, rect, cfg, acquisition_ms, *, pixel_um=3.45):
        started = time.perf_counter()
        self.sequence += 1
        result = dict(sequence=self.sequence, timestamp=time.time(), valid=False,
                      acquisition_ms=acquisition_ms)
        try:
            if rect is None:
                raise ValueError('Define the green rectangle')
            w,h,ox,oy = roi
            x1,y1,x2,y2 = rect
            if not (ox <= x1 < x2 <= ox+w and oy <= y1 < y2 <= oy+h):
                raise ValueError('Camera ROI must contain the entire green rectangle')
            region, bounds = crop_rect(frame, roi, rect)
            spots = detect_spots(region, max_pixels=4096 * 4096, cfg=cfg)
            result['detected_spots'] = len(spots)
            result['peak_raw'] = int(region.max())
            result.update(response_from_centers(
                spots.x + x1, pixel_um=pixel_um, focal_mm=cfg.angular_focal_mm))
            result.update(pixel_um=pixel_um, focal_mm=cfg.angular_focal_mm)
            result['valid'] = True
        except ValueError as exc:
            result['error'] = str(exc)
        now = time.perf_counter()
        result['analysis_ms'] = (now-started)*1000
        result['frame_interval_ms'] = None if self.previous_time is None else (now-self.previous_time)*1000
        self.previous_time = now
        self.latest = result
