"""Camera discovery with optional vendor SDKs and an explicit simulation mode."""
from __future__ import annotations

import logging

from .base import Camera
from .simulated import SimulatedCamera

__all__ = ["Camera", "SimulatedCamera", "open_cameras"]


def _open_basler(mode):
    from pypylon import pylon
    from .basler import BaslerCamera
    cameras = []
    try:
        for i, _ in enumerate(pylon.TlFactory.GetInstance().EnumerateDevices()):
            cameras.append(BaslerCamera(mode=mode, device_idx=i))
        return cameras
    except Exception:
        for camera in cameras:
            camera.close()
        raise


def open_cameras(mode="16Bit", simulate=False, backend="auto", **sim_opts):
    if backend not in ("auto", "flir", "basler", "sim"):
        raise ValueError(f"Unknown camera backend: {backend}")
    if simulate or backend == "sim":
        return [SimulatedCamera(mode=mode, **sim_opts)]
    from .flir import open_flir_cameras
    cameras = []
    diagnostics = []
    for name, discover in (("flir", open_flir_cameras), ("basler", _open_basler)):
        if backend not in ("auto", name):
            continue
        try:
            found = discover(mode)
            if not found:
                diagnostics.append(f"{name.upper()}: no cameras detected")
            if not found and backend == name:
                raise RuntimeError(f"No {name.upper()} cameras detected. Check USB and the vendor viewer.")
            cameras.extend(found)
        except Exception as exc:
            if backend == name:
                raise RuntimeError(f"Cannot open {name.upper()} camera: {exc}") from exc
            logging.warning("%s discovery: %s", name.upper(), exc)
            diagnostics.append(f"{name.upper()}: {exc}")
    if cameras:
        return cameras
    raise RuntimeError(
        "No camera could be opened.\n\n" + "\n".join(diagnostics)
        + "\n\nCheck the camera connection and any macOS accessory prompt, then reopen the app."
        + "\nSimulation is available only when explicitly selected with --sim."
    )
