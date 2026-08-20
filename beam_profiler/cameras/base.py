"""Camera interface shared by the Basler driver and the simulator."""

from __future__ import annotations

from collections import deque

import numpy as np


class AutoExposure:
    """Keeps the frame maximum near saturation: coarse 10% steps far from the
    target, PID fine-tuning inside the tolerance band."""

    def __init__(self, kp=0.05, ki=0.0, kd=0.3, target=0.8, tolerance=0.19):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target_frac, self.tol_frac = target, tolerance
        self._buffer: deque = deque(maxlen=5)
        self._integral = 0.0
        self._last_error = 0.0

    def step(self, max_val: float, exposure: float, saturation: int) -> float:
        """Return the next exposure time given the current frame maximum."""
        target = self.target_frac * saturation
        tol = self.tol_frac * saturation
        self._buffer.append(max_val)
        if abs(max_val - target) > tol:
            self._integral = self._last_error = 0.0
            return exposure * (1.1 if max_val < target else 0.9)
        error = (target - np.mean(self._buffer)) / saturation
        self._integral += error
        derivative = error - self._last_error
        self._last_error = error
        control = self.kp * error + self.ki * self._integral + self.kd * derivative
        return exposure * (1 + control)


class Camera:
    """Base class: subclasses implement _grab() and the ROI / ExposureTime /
    Gain properties, and set mode, model_name, serial, W, H, pixel_size,
    default_roi in __init__."""

    mode: str
    model_name: str
    serial: str
    W: int
    H: int
    pixel_size: float
    default_roi: tuple

    def __init__(self, mode: str = "16Bit"):
        if mode not in ("8Bit", "16Bit"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.auto_exposure = False
        self.min_exposure = 20
        self.max_exposure = 20000
        self._ae = AutoExposure()
        self.userdefined_line_available = False
        self.userdefined_line_enabled = False

    @property
    def saturation(self) -> int:
        return 255 if self.mode == "8Bit" else 65535

    def AutoExposureOn(self, range: tuple[float, float] | None = None):
        self.auto_exposure = True
        if range is not None:
            self.min_exposure, self.max_exposure = range

    def AutoExposureOff(self):
        self.auto_exposure = False

    def grab_image(self) -> np.ndarray | None:
        img = self._grab()
        if img is not None and self.auto_exposure:
            new_exp = self._ae.step(float(img.max()), self.ExposureTime, self.saturation)
            self.ExposureTime = float(np.clip(new_exp, self.min_exposure, self.max_exposure))
        return img

    def set_userdefined_line(self, enable: bool) -> bool:
        """Hardware sync output; only real cameras support this."""
        return False

    # --- to implement in subclasses -------------------------------------- #
    def _grab(self) -> np.ndarray | None:
        raise NotImplementedError

    ROI: tuple  # (w, h, ox, oy) property
    ExposureTime: float  # µs property
    Gain: float  # property

    def close(self):
        pass
