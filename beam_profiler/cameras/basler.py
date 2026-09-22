"""pypylon driver for Basler cameras."""

from __future__ import annotations

import logging

import numpy as np
from pypylon import pylon

from ..config import load_camera_config
from .base import Camera

CAMERA_CONFIG = load_camera_config()


class BaslerCamera(Camera):
    def __init__(self, mode: str = "16Bit", device_idx: int = 0):
        super().__init__(mode)
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if device_idx >= len(devices):
            raise IndexError(f"Only {len(devices)} cameras found, idx={device_idx}")
        self.device_info = devices[device_idx]
        self.camera = pylon.InstantCamera(factory.CreateDevice(self.device_info))
        self.model_name = self.device_info.GetModelName()
        self.serial = self.device_info.GetSerialNumber()
        logging.info(f"Connected to camera: {self.model_name}")

        settings = CAMERA_CONFIG[self.model_name]
        self.default_roi = settings["default_roi"]
        self.pixel_size = settings["pixel_size"]

        self.camera.Open()
        self.W = int(self.camera.Width.Max)
        self.H = int(self.camera.Height.Max)

        self.converter = pylon.ImageFormatConverter()
        if mode == "16Bit":
            logging.info("Camera operating in 12-bit mode")
            self.camera.PixelFormat.Value = "Mono12"
            self.converter.OutputPixelFormat = pylon.PixelType_Mono16
        else:
            logging.info("Camera operating in 8-bit mode")
            self.camera.PixelFormat.Value = "Mono8"
            self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Software triggering.  Not every model exposes it - grab only fires the
        # trigger if it was armed, otherwise the camera free-runs.
        self.software_trigger = False
        try:
            self.camera.TriggerSelector.Value = "FrameStart"
            self.camera.TriggerMode.Value = "On"
            self.camera.TriggerSource.Value = "Software"
            self.software_trigger = True
        except Exception as e:
            logging.warning(f"Software trigger unavailable, free-running: {e}")

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.Gain = 0
        self.ExposureTime = 200

        # Line2 mirrors the exposure window (active-low) for scope / DAQ sync.
        self.exposure_line_available = False
        try:
            self.camera.LineSelector.Value = "Line2"
            self.camera.LineMode.Value = "Output"
            self.camera.LineInverter.Value = True
            self.camera.LineSource.Value = "ExposureActive"
            self.exposure_line_available = True
        except Exception as e:
            logging.warning(f"Line2 exposure output unavailable: {e}")

        # Line3 is a user-driven output, pulsed low around each acquisition.
        self.userdefined_line_available = self.set_userdefined_line(False)
        if not self.userdefined_line_available:
            logging.warning("Line3 user output unavailable on this camera")

    def set_userdefined_line(self, enable: bool) -> bool:
        try:
            self.camera.LineSelector.Value = "Line3"
            if enable:
                self.camera.LineMode.Value = "Output"
                self.camera.LineInverter.Value = False
                self.camera.LineSource.Value = "UserOutput1"
                self.camera.UserOutputSelector.Value = "UserOutput1"
                self.camera.UserOutputValue.Value = True
            else:
                self.camera.LineMode.Value = "Input"
        except Exception as e:
            logging.error(f"Failed to set user-defined line: {e}")
            self.userdefined_line_enabled = False
            return False
        self.userdefined_line_enabled = enable
        return True

    def _set_user_output(self, high: bool):
        self.camera.UserOutputSelector.Value = "UserOutput1"
        self.camera.UserOutputValue.Value = high

    # ------------------------------ properties --------------------------- #
    @property
    def Gain(self):
        return self.camera.Gain.Value

    @Gain.setter
    def Gain(self, value):
        self.camera.Gain.Value = value

    @property
    def ExposureTime(self):
        return self.camera.ExposureTime.Value

    @ExposureTime.setter
    def ExposureTime(self, value):
        self.camera.ExposureTime.Value = float(
            np.clip(value, self.min_exposure, self.max_exposure)
        )

    @property
    def roi_constraints(self):
        return tuple((int(n.Min), int(n.Inc)) for n in
                     (self.camera.Width, self.camera.Height,
                      self.camera.OffsetX, self.camera.OffsetY))

    @property
    def ROI(self):
        return (
            self.camera.Width.Value,
            self.camera.Height.Value,
            self.camera.OffsetX.Value,
            self.camera.OffsetY.Value,
        )

    @ROI.setter
    def ROI(self, value):
        if not self.camera.IsOpen():
            logging.error("Camera is not open, cannot set ROI.")
            return
        try:
            was_grabbing = self.camera.IsGrabbing()
            if was_grabbing:
                self.camera.StopGrabbing()
            self._apply_roi(*value)
            if was_grabbing:
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            logging.error(f"Failed to set ROI: {e}")

    def _apply_roi(self, width, height, offset_x, offset_y):
        width = max(10, min(width, self.W))
        height = max(10, min(height, self.H))
        offset_x = min(offset_x, self.W - width)
        offset_y = min(offset_y, self.H - height)
        self.camera.Width.Value = int(self._align(width, self.camera.Width))
        self.camera.Height.Value = int(self._align(height, self.camera.Height))
        self.camera.OffsetX.Value = int(self._align(offset_x, self.camera.OffsetX))
        self.camera.OffsetY.Value = int(self._align(offset_y, self.camera.OffsetY))

    @staticmethod
    def _align(value, node):
        """Floor `value` to the node's nearest valid increment."""
        return value - (value - node.Min) % node.Inc

    # ------------------------------ acquisition -------------------------- #
    def _grab(self) -> np.ndarray | None:
        if self.userdefined_line_enabled:
            self._set_user_output(False)  # pulse low around the acquisition
        if self.software_trigger:
            self.camera.TriggerSoftware.Execute()
        result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if self.userdefined_line_enabled:
            self._set_user_output(True)
        if not result.GrabSucceeded():
            result.Release()
            return None
        img = self.converter.Convert(result).GetArray().copy()
        result.Release()
        return img

    def close(self):
        self.camera.StopGrabbing()
        self.camera.Close()
