"""Empirical flat-field correction in original sensor/container units."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


def display_gray(frame, full_scale):
    """Linear display scaling, including floating corrected frames and negatives."""
    scaled = np.array(frame, dtype=np.float32, copy=True)
    scaled *= 256.0 / (full_scale + 1)
    np.nan_to_num(scaled, copy=False, nan=0, posinf=255, neginf=0)
    np.clip(scaled, 0, 255, out=scaled)
    np.rint(scaled, out=scaled)
    return scaled.astype(np.uint8)


@dataclass
class FlatField:
    gain: np.ndarray
    dark: np.ndarray
    path: str = ''
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.gain = np.asarray(self.gain, dtype=np.float32)
        self.dark = np.asarray(self.dark, dtype=np.float32)
        if self.gain.ndim != 2 or self.dark.shape != self.gain.shape:
            raise ValueError('Gain and dark maps must have the same 2-D sensor shape')
        self.valid = np.isfinite(self.gain) & (self.gain > 0) & np.isfinite(self.dark)
        if not self.valid.any():
            raise ValueError('Calibration contains no valid pixels')
        self.invalid_pixels = int(self.valid.size - np.count_nonzero(self.valid))

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            summary = path / 'summary.json'
            meta = json.loads(summary.read_text()) if summary.exists() else {}
            before = meta.get('camera_before', {})
            config = dict(camera_serial=meta.get('camera_serial'), exposure_us=meta.get('exposure_us'),
                          gain_db=before.get('Gain'), adc_bit_depth=before.get('AdcBitDepth'),
                          pixel_format=before.get('PixelFormat'), gain_map='gain_map.npy',
                          dark_map=meta.get('dark_map', 'dark_mean_6.npy'))
            base = path
        else:
            config = json.loads(path.read_text())
            base = path.parent
        gain = np.load(base / config['gain_map'], mmap_mode='r', allow_pickle=False)
        dark = np.load(base / config['dark_map'], mmap_mode='r', allow_pickle=False)
        return cls(gain, dark, str(path), config)

    def mismatch(self, camera):
        serial = self.metadata.get('camera_serial')
        if serial and str(camera.serial) != str(serial):
            return 'different camera serial'
        if self.gain.shape != (camera.H, camera.W):
            return 'different full-sensor dimensions'
        expected_format = self.metadata.get('pixel_format')
        if expected_format and expected_format != ('Mono16' if camera.mode == '16Bit' else 'Mono8'):
            return 'different pixel format'
        for key, value in [('exposure_us', camera.ExposureTime), ('gain_db', camera.Gain)]:
            expected = self.metadata.get(key)
            if expected is not None and not np.isclose(float(value), float(expected), rtol=1e-6, atol=1e-6):
                return f'different {key} (expected {expected})'
        return None

    def check_adc(self, camera):
        """Read once on startup; never change camera configuration for a map."""
        expected = self.metadata.get('adc_bit_depth')
        if expected and hasattr(camera, '_node'):
            actual = camera._node('AdcBitDepth', 'Enumeration').ToString()
            if actual != expected:
                raise ValueError(f'Calibration expects {expected} ADC, camera reports {actual}')

    def apply(self, frame, roi):
        w, h, ox, oy = map(int, roi)
        if frame.shape != (h, w) or min(ox, oy) < 0 or ox+w > self.gain.shape[1] or oy+h > self.gain.shape[0]:
            raise ValueError('Camera ROI is outside the calibrated sensor')
        region = np.s_[oy:oy+h, ox:ox+w]
        result = np.empty((h, w), np.float32)
        # Invalid calibration pixels are explicitly masked to zero, never infinities.
        np.subtract(frame, self.dark[region], out=result)
        np.divide(result, self.gain[region], out=result, where=self.valid[region])
        if self.invalid_pixels:
            result[~self.valid[region]] = 0
        return result


class FrameAverage:
    """Streaming arithmetic mean and sample variance; no stored raw stack."""
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def add(self, frame):
        if self.mean is None:
            self.mean = np.zeros(frame.shape, np.float64)
            self.m2 = np.zeros(frame.shape, np.float64)
        if frame.shape != self.mean.shape:
            raise ValueError('Frame shape changed during averaging')
        self.n += 1
        delta = frame - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (frame - self.mean)

    @property
    def variance(self):
        return self.m2 / max(self.n - 1, 1)
