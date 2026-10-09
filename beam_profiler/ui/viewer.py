"""Interactive viewer: main loop, keyboard/mouse handling and overlay layout.

Keyboard shortcuts are listed in the README."""

from __future__ import annotations

import logging
from collections import deque
import os
import sys
import time
from threading import Thread

import cv2
import numpy as np

from ..config import (
    DATA_DIR,
    FPS_LIMIT,
    MAX_EXPOSURE_US,
    MIN_EXPOSURE_US,
    PAD_COLOR,
    WEB_SERVER_PORT,
    WINDOW_SIZE,
)
from .. import fitconfig
from ..processing import SpotArray, gpu
from ..processing.pipeline import FrameProcessor, AnalysisOptions
from ..processing.displacement import ImageGridAverage
from ..session import LiveSession, SessionSnapshot
from ..roi import ROIModel, ViewTransform
from ..status import write_status, viewer_status
from . import overlays, settings

WINDOW = "Beam Profiler - Basler / FLIR"

# waitKeyEx arrow codes: Windows, Cocoa(macOS), and Linux(GTK) report different values
KEY_FACTORS = {
    2555904: 1.1, 65363: 1.1, 63235: 1.1,  # right: +10%
    2424832: 0.9, 65361: 0.9, 63234: 0.9,  # left:  -10%
    2490368: 10, 65362: 10, 63232: 10,  # up:    x10
    2621440: 0.1, 65364: 0.1, 63233: 0.1,  # down:  /10
}


class Viewer:
    def __init__(self, cameras: list):
        self._quit_requested = False
        self.camera_failed = False
        self._native_close_handler = None
        self._native_key_monitor = None
        self._pending_keys = deque()
        self.cameras = cameras
        self.curr_idx = 0
        self.camera = cameras[0]
        self.roi_model = ROIModel(self.camera.W, self.camera.H)
        self.camera.ROI = self.roi_model.tuple
        self._cached_state: dict[str, dict] = {}  # per-camera, for seamless switching

        self._auto_exp = False
        self.do_fitting = False
        self.profiler_enabled = False
        self.profiler_message = ''
        self.show_stats = False
        self.row_col_fitting = False
        self.show_displacement = False
        self._displacement_average = ImageGridAverage()
        self.exposure_us = 200
        self.last_mouse = (0, 0)  # sensor coords
        self._mouse_display = None
        self._pending_scroll = [0.0, 0.0]
        self.rect_disp = None  # white rectangle while dragging (display coords)
        self.rect_sensor = None
        self.fit_rect_disp = None  # green fitting rectangle while dragging
        self.fit_rect_sensor = None

        self.avg_dt = 1 / FPS_LIMIT
        self.processor = FrameProcessor()
        self.statistics = self.processor.statistics
        self.analysis = None
        self.session = LiveSession()

        # Current display state; HTTP consumers use the published session snapshot.
        self.current_frame = None
        self.latest_spots = SpotArray.empty()
        self.current_rect_stats = None
        self.web_server_enabled = False
        self.show_rect_stats = False
        self._web_thread = None
        self.angular_monitor = None  # opt-in single-shot dispersion view

        # WINDOW_GUI_NORMAL drops the Qt toolbar and the pixel-hover overlay,
        # which render as black/garbled boxes (OpenCV's bundled Qt has no fonts)
        gpu.warm_up()  # pay the CuPy import off the critical path
        settings.prepare_gui()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW, WINDOW_SIZE, WINDOW_SIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        if sys.platform == "darwin":
            from .macos import attach_close_button, attach_key_monitor
            self._native_close_handler = attach_close_button(WINDOW, self.request_close)
            self._native_key_monitor = attach_key_monitor(WINDOW, self._pending_keys.append)

    def request_close(self):
        self._quit_requested = True


    # ------------------------------ properties --------------------------- #
    @property
    def auto_exp(self):
        return self._auto_exp

    @auto_exp.setter
    def auto_exp(self, value):
        self._auto_exp = value
        self._exposure_request = None
        if value:
            self.camera.AutoExposureOn(range=(MIN_EXPOSURE_US, MAX_EXPOSURE_US))
        else:
            self.camera.AutoExposureOff()
            self.exposure_us = int(np.clip(self.exposure_us, MIN_EXPOSURE_US, MAX_EXPOSURE_US))

    @property
    def view(self) -> ViewTransform:
        return ViewTransform(self.camera.ROI, WINDOW_SIZE)

    # ------------------------------ main loop ----------------------------- #
    def run(self, max_frames=None):
        try:
            self._run_loop(max_frames=max_frames)
        finally:
            # Cleanup every GUI resource even if one toolkit reports an error.
            try:
                if self._native_key_monitor is not None:
                    from .macos import remove_key_monitor
                    remove_key_monitor(self._native_key_monitor)
                    self._native_key_monitor = None
            finally:
                try:
                    cv2.destroyAllWindows()
                finally:
                    settings.destroy_settings()

    def _run_loop(self, max_frames=None):
        frame_count = 0
        os.makedirs(DATA_DIR, exist_ok=True)
        while True:
            settings.pump_events()
            if self._quit_requested or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            self._apply_web_commands()
            self._apply_scroll()
            t0 = time.time()
            grab_started = time.perf_counter()
            frame = self._grab_frame()
            acquisition_ms = (time.perf_counter() - grab_started) * 1000
            if frame is None:
                if self._quit_requested:
                    break
                if not self._poll_keys(10):
                    break
                continue
            self.current_frame = frame
            if self.angular_monitor is not None:
                self.angular_monitor.update(
                    frame, self.camera.ROI, self.fit_rect_sensor,
                    fitconfig.active(), acquisition_ms, pixel_um=self.camera.pixel_size * 1e6,
                )
            frame_disp = (
                cv2.convertScaleAbs(frame, alpha=1 / 256.0)
                if self.camera.mode == "16Bit"
                else frame
            )
            self.frame_disp = frame_disp

            self._analyze_frame(frame)
            self.session.publish(SessionSnapshot(
                result=self.analysis, status=viewer_status(self),
                angular=self.angular_monitor.latest if self.angular_monitor is not None else None,
            ))

            disp = self._compose(frame_disp, self.latest_spots)
            cv2.imshow(WINDOW, disp)
            if self.angular_monitor is not None:
                self._spacing_window.update(self.angular_monitor.latest)
            if not self._poll_keys(1):
                break
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
            self._sleep_for_fps(time.time() - t0)

    def _camera_unavailable(self, message):
        self.camera_failed = True
        self.current_frame = None
        self.analysis = None
        self.latest_spots = SpotArray.empty()
        status = dict(self.session.read().status)
        status.update(camera_available=False, error=message)
        self.session.publish(SessionSnapshot(status=status))
        logging.error("%s; closing Beam Profiler", message)
        self._quit_requested = True

    def _grab_frame(self):
        try:
            frame = self.camera.grab_image()
        except Exception as exc:
            self._camera_unavailable(f'Camera acquisition failed: {exc}')
            return None
        if frame is None:
            self._camera_unavailable('Camera stopped delivering frames')
        return frame

    def _analyze_frame(self, frame):
        self.analysis = self.processor.process(
            frame, roi=self.camera.ROI, pixel_size=self.camera.pixel_size,
            full_scale=self.camera.saturation,
            options=AnalysisOptions(
                fitting=self.do_fitting, profiler=self.profiler_enabled,
                fit_rect=self.fit_rect_sensor, rect=self.rect_sensor,
                rect_stats=self.show_rect_stats,
            ),
        )
        self.current_frame = self.analysis.frame
        self.latest_spots = self.analysis.spots
        self.statistics = self.analysis.statistics
        self.current_rect_stats = self.analysis.rect_stats
        self.profiler_message = self.analysis.profiler_message
        if self.show_displacement and not self.profiler_enabled:
            self._displacement_average.update(
                frame, roi=self.camera.ROI, camera=self.camera.serial,
                exposure=self.camera.ExposureTime, gain=self.camera.Gain,
                full_scale=self.camera.saturation, config=fitconfig.active(),
                fit_rect=self.fit_rect_sensor, frame_id=self.analysis.timestamp)
        else:
            self._displacement_average.reset()

    def _apply_web_commands(self):
        for name, value in self.session.take_commands():
            if name == "fit_config":
                fitconfig.set_active(fitconfig.active().replace(**value))
            else:
                setattr(self, name, value)

    def _compose(self, frame_disp, spots) -> np.ndarray:
        """Resize + pad the frame to the window and draw all overlays.

        Everything is drawn in display space after the resize, so text and
        line widths do not scale with the ROI zoom."""
        view = self.view
        resized = cv2.resize(frame_disp, (view.view_w, view.view_h))
        # Color only the displayed intensity; fitting and saved frames stay raw.
        colored = (cv2.applyColorMap(resized, cv2.COLORMAP_TURBO)
                   if fitconfig.active().heatmap
                   else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))
        disp = cv2.copyMakeBorder(
            colored,
            view.pad_t,
            WINDOW_SIZE - view.view_h - view.pad_t,
            view.pad_l,
            WINDOW_SIZE - view.view_w - view.pad_l,
            cv2.BORDER_CONSTANT,
            value=PAD_COLOR,
        )
        saturated = self.analysis.saturated
        self._draw_rectangles(disp, view)
        if self.row_col_fitting and self.do_fitting and not self.profiler_enabled:
            overlays.draw_row_col(disp, view, self.statistics.rows, self.statistics.columns,
                                  self.statistics.grid_shear)
        displacement = None
        if self.show_displacement and not self.profiler_enabled:
            if not hasattr(self, '_displacement_average'):
                self._displacement_average = ImageGridAverage()
            displacement = self._displacement_average.latest
            overlays.draw_displacement(disp, view, displacement)
        elif hasattr(self, '_displacement_average'):
            self._displacement_average.reset()
        if self.show_displacement and not self.profiler_enabled and not self.do_fitting:
            averaged_spots = self._displacement_average.spots
            if averaged_spots is not None:
                spots = averaged_spots
                saturated = np.zeros(len(spots), dtype=bool)
        # Draw the actual beam fits above grid/displacement lines, in every mode.
        if len(spots):
            overlays.draw_spots(disp, spots, view, self.camera.pixel_size, saturated=saturated,
                                show_centers=self.row_col_fitting or self.show_displacement)

        y = overlays.draw_hud(disp, self._hud_lines())
        if self.show_rect_stats:
            y = overlays.draw_rect_stats(disp, self.current_rect_stats, y)
        if self.row_col_fitting and not self.profiler_enabled:
            y = overlays.draw_grid_shear(disp, self.statistics.grid_shear, y)
        if displacement is not None:
            y = overlays.draw_displacement_stats(disp, displacement, y)
        if self.show_stats and not self.profiler_enabled:
            bar_y = overlays.draw_spot_stats(
                disp,
                y,
                self.statistics.std_dx_ema,
                self.statistics.std_dy_ema,
                float(np.mean(self.statistics.stats_dx)) * self.camera.pixel_size * 1e6,
                float(np.mean(self.statistics.stats_dy)) * self.camera.pixel_size * 1e6,
                self.statistics.curvature(self.statistics.stats_dx),
                self.statistics.curvature(self.statistics.stats_dy),
                self.statistics.sigma_std_ema,
                show_curvature=not self.row_col_fitting and not self.statistics._stats_grid_mode,
            )
            if self.row_col_fitting:
                overlays.draw_grid_stats(
                    disp, bar_y, self.statistics.rows, self.statistics.columns, self.statistics.grid_stats,
                    self.camera.pixel_size * 1e6,
                )
        if np.any(saturated):
            overlays.text(disp, f"SATURATED: {np.count_nonzero(saturated)} beam(s) - reduce exposure; fit unreliable",
                          (10, WINDOW_SIZE - 18), (0, 0, 255), scale=0.65)
        return disp

    def _hover_pixel(self):
        """Read the raw sensor pixel, never the resized display or its overlays."""
        if self.current_frame is None or self._mouse_display is None:
            return None
        view = self.view
        if not view.contains(*self._mouse_display):
            return None
        sx, sy = (int(np.floor(v + 0.5)) for v in view.to_sensor(*self._mouse_display))
        x, y = sx - view.ox, sy - view.oy
        h, w = self.current_frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        return sx, sy, int(self.current_frame[y, x])

    def _hud_lines(self) -> list[str]:
        exposure = self.camera.ExposureTime
        exposure_txt = f"Auto = {exposure:.0f}" if self.auto_exp else f"{exposure}"
        um = self.camera.pixel_size * 1e6
        rect_w = (self.rect_sensor[2] - self.rect_sensor[0]) * um if self.rect_sensor else 0
        rect_h = (self.rect_sensor[3] - self.rect_sensor[1]) * um if self.rect_sensor else 0
        hover = self._hover_pixel()
        mouse_line = (f'Mouse: ({hover[0]}, {hover[1]})  Raw: {hover[2]} DN'
                      if hover is not None else 'Mouse: --  Raw: --')
        fitting_line = (f'Profiler: ON (cpu) - {self.profiler_message} (P to toggle)'
                        if self.profiler_enabled else
                        f'Fitting: {self.do_fitting} ({gpu.backend_name()}), Stats: {self.show_stats}, '
                        f'Row-Col: {self.row_col_fitting} (P: single beam)')
        detail = f"Fit: {fitconfig.describe()} (Press 'o' for settings)"
        if self.profiler_enabled:
            detail = 'Single Gaussian + background; Shift-drag selects region; v clears'
            if len(self.latest_spots):
                spot = self.latest_spots[0]
                detail = (f"1/e^2 radii: {spot['sigma_0']*um:.1f}, {spot['sigma_1']*um:.1f} um; "
                          'Shift-drag region, v clears')
        return [
            f"Camera: {self.camera.model_name}, FPS: {1 / self.avg_dt:4.1f}",
            f"ROI: {self.camera.ROI}  {mouse_line}",
            f"Exposure: {exposure_txt} us",
            f"Rect: {self.rect_sensor}, W = {rect_w:.1f} um, H = {rect_h:.1f} um",
            fitting_line,
            detail,
            f"Web Server: {'ON' if self.web_server_enabled else 'OFF'}, "
            f"Rect Stats: {'ON' if self.show_rect_stats else 'OFF'} (Press 'w' to toggle)",
            f"User output: {self.camera.userdefined_line_enabled} (Press 'y' to toggle)",
        ]

    def _draw_rectangles(self, disp, view: ViewTransform):
        for rect, color in ((self.rect_disp, (255, 255, 255)), (self.fit_rect_disp, (0, 255, 0))):
            if rect is not None:
                cv2.rectangle(disp, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
        for rect, color in (
            (self.rect_sensor, (255, 255, 255)),
            (self.fit_rect_sensor, (0, 255, 0)),
        ):
            if rect is not None:
                p1 = view.to_display(rect[0], rect[1])
                p2 = view.to_display(rect[2], rect[3])
                cv2.rectangle(disp, p1, p2, color, 2)

    # ------------------------------ statistics ---------------------------- #
    # ------------------------------ mouse --------------------------------- #
    def _apply_scroll(self):
        zoom, aspect = self._pending_scroll
        self._pending_scroll = [0.0, 0.0]
        if not (zoom or aspect) or self._mouse_display is None:
            return
        view = self.view
        if not view.contains(*self._mouse_display):
            return
        actual = tuple(self.camera.ROI)
        model = self.roi_model
        now = time.monotonic()
        # Keep fractional zoom requests rather than discarding them whenever
        # the camera rounds a size to its hardware increment.
        if (getattr(self, '_scroll_camera', None) is not self.camera
                or getattr(self, '_scroll_roi', None) != actual):
            w, h, ox, oy = actual
            model.scale, model.aspect = w / model.full_w, w / h
            model.cx, model.cy = ox + w / 2, oy + h / 2
            self._scroll_anchor = None
        if (getattr(self, '_scroll_anchor', None) is None
                or getattr(self, '_scroll_pointer', None) != self._mouse_display
                or now - getattr(self, '_scroll_time', 0) > 0.3):
            self._scroll_anchor = view.to_sensor(*self._mouse_display)
        anchor = self._scroll_anchor
        if aspect:
            model.keep_point_fixed(
                *anchor, new_aspect=model.aspect * np.exp(-np.clip(aspect, -0.35, 0.35)))
        if zoom:
            model.scale = float(np.clip(model.scale * np.exp(-np.clip(zoom, -0.35, 0.35)), .02, 1.0))
        limits = getattr(self.camera, 'roi_constraints', ((1, 1), (1, 1), (0, 1), (0, 1)))

        def aligned(value, limit, maximum):
            minimum, increment = limit
            last = minimum + ((maximum - minimum) // increment) * increment
            return int(np.clip(minimum + np.rint((value - minimum) / increment) * increment,
                               minimum, last))

        # Align dimensions FIRST, then anchor using the actual display mapping.
        # Anchoring before camera rounding moves the beam at high magnification.
        w, h = model.size
        w = aligned(w, limits[0], model.full_w)
        h = aligned(h, limits[1], model.full_h)
        target_view = ViewTransform((w, h, 0, 0), WINDOW_SIZE)
        local = target_view.to_sensor(*self._mouse_display)
        ox = aligned(anchor[0] - local[0], limits[2], model.full_w - w)
        oy = aligned(anchor[1] - local[1], limits[3], model.full_h - h)
        requested = (w, h, ox, oy)
        if requested != actual:
            self.camera.ROI = requested
            self.current_frame = None
        self._scroll_roi = tuple(self.camera.ROI)
        self._scroll_camera = self.camera
        self._scroll_pointer = self._mouse_display
        self._scroll_time = now

    def on_mouse(self, event, x, y, flags, _param):
        if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            if sys.platform == "darwin":
                # Cocoa supplies scroll deltas as x/y, NOT cursor coordinates;
                # flags contain modifiers only. Keep the last real pointer.
                delta = y if event == cv2.EVENT_MOUSEWHEEL else x
                amount = delta * 0.005
            else:
                if event == cv2.EVENT_MOUSEHWHEEL:
                    return
                self._mouse_display = (x, y)
                delta = (flags >> 16) & 0xffff
                if delta >= 0x8000:
                    delta -= 0x10000
                amount = delta / 120 * 0.1
            aspect = bool(flags & (cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_SHIFTKEY))
            if event == cv2.EVENT_MOUSEHWHEEL and not aspect:
                return
            self._pending_scroll[int(aspect)] += amount
            return

        self._mouse_display = (x, y)
        view = self.view
        if view.contains(x, y):
            sx, sy = view.to_sensor(x, y)
            self.last_mouse = (int(np.floor(sx + 0.5)), int(np.floor(sy + 0.5)))
        ctrl = flags & (cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_SHIFTKEY)

        if event == cv2.EVENT_LBUTTONDOWN and view.contains(x, y):
            if ctrl:
                self.fit_rect_disp = [x, y, x, y]
            else:
                self.rect_disp = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.fit_rect_disp is not None:
                self.fit_rect_disp[2:] = [x, y]
            elif self.rect_disp is not None:
                self.rect_disp[2:] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.fit_rect_disp is not None:
                self.fit_rect_sensor = self._finish_rect(self.fit_rect_disp, view)
                self.fit_rect_disp = None
                write_status(self)
            elif self.rect_disp is not None:
                self.rect_sensor = self._finish_rect(self.rect_disp, view)
                self.rect_disp = None
                write_status(self)

    @staticmethod
    def _finish_rect(rect_disp, view: ViewTransform) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect_disp
        s1 = view.to_sensor(min(x1, x2), min(y1, y2))
        s2 = view.to_sensor(max(x1, x2), max(y1, y2))
        return (int(s1[0]), int(s1[1]), int(s2[0]), int(s2[1]))

    # ------------------------------ keys ---------------------------------- #
    def _toggle_angular(self):
        if self.angular_monitor is not None:
            self.angular_monitor = None
            self._spacing_window.destroy()
            self._spacing_window = None
            return
        from ..angular import AngularMonitor
        from .spacing import SpacingWindow
        self._spacing_window = SpacingWindow(self._toggle_angular)
        self.angular_monitor = AngularMonitor()

    def _handle_key(self, key: int) -> bool:
        if ord("A") <= key <= ord("Z"):
            key += ord("a") - ord("A")
        if key in (ord('a'), ord('A')):
            self._toggle_angular()
            return True
        if key == 27:  # ESC
            return False
        if key in KEY_FACTORS:
            self._adjust_exposure(KEY_FACTORS[key])
        elif key == ord("a"):
            self.auto_exp = not self.auto_exp
        elif key in (ord('p'), ord('P')):
            self.profiler_enabled = not self.profiler_enabled
            self.profiler_message = ''
            self.latest_spots = SpotArray.empty()
        elif key in (ord('f'), ord('F')):
            # F selects the array fitter even when P currently overrides it.
            self.do_fitting = True if self.profiler_enabled else not self.do_fitting
            self.profiler_enabled = False
            self.profiler_message = ''
            self.latest_spots = SpotArray.empty()
        elif key == ord("g"):
            self.show_stats = not self.show_stats
        elif key == ord("h"):
            self.row_col_fitting = not self.row_col_fitting
        elif key == ord("e"):
            self.show_displacement = not self.show_displacement
            if hasattr(self, '_displacement_average'):
                self._displacement_average.reset()
        elif key == ord("c"):
            self.rect_sensor = None
        elif key == ord("v"):
            self.fit_rect_sensor = None
        elif key == ord("s"):
            self._quick_save()
        elif key == ord("d"):
            self._dialog_save()
        elif key == ord("t"):
            self._switch_camera()
        elif key == ord("w"):
            self._toggle_web_server()
        elif key == ord("o"):
            settings.open_settings()
        elif key == ord("y"):
            self.camera.set_userdefined_line(not self.camera.userdefined_line_enabled)
            logging.info(
                f"userdefined line sync mode set to {self.camera.userdefined_line_enabled}"
            )
        if key != -1:
            write_status(self)
        return True

    def _poll_keys(self, delay):
        # Cocoa waitKey can consume events belonging to Tk entry fields.
        # Tk already pumps the shared event queue at the top of every frame.
        if sys.platform == 'darwin':
            from .macos import viewer_has_keyboard_focus
            key = cv2.waitKeyEx(delay) if viewer_has_keyboard_focus(WINDOW) else -1
        else:
            key = cv2.waitKeyEx(delay)
        if key != -1:
            self._pending_keys.append(key)
        while self._pending_keys:
            if not self._handle_key(self._pending_keys.popleft()):
                return False
        return True

    def _adjust_exposure(self, factor: float):
        if self.auto_exp:
            return
        actual = float(self.camera.ExposureTime)
        direction = 1 if factor > 1 else -1
        target = getattr(self, '_exposure_request', None)
        if (target is None or getattr(self, '_exposure_camera', None) is not self.camera
                or getattr(self, '_exposure_readback', None) != actual
                or getattr(self, '_exposure_direction', None) != direction):
            target = actual
        minimum = max(MIN_EXPOSURE_US, getattr(self.camera, 'min_exposure', MIN_EXPOSURE_US))
        maximum = min(MAX_EXPOSURE_US, getattr(self.camera, 'max_exposure', MAX_EXPOSURE_US))
        # Keep fractional requests: hardware may round several successive
        # requests to the same exposure, especially near its minimum.
        target = float(np.clip(target * factor, minimum, maximum))
        self.camera.ExposureTime = target
        self.exposure_us = float(self.camera.ExposureTime)
        self._exposure_request = target
        self._exposure_readback = self.exposure_us
        self._exposure_camera = self.camera
        self._exposure_direction = direction

    # ------------------------------ actions ------------------------------- #
    def _save_frame(self, jpg_path: str):
        """JPEG of the display frame plus the raw frame as .npy."""
        cv2.imwrite(jpg_path, self.frame_disp)
        np.save(os.path.splitext(jpg_path)[0] + ".npy", self.current_frame)
        logging.info("Saved %s (+ .npy)", jpg_path)

    def _quick_save(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._save_frame(os.path.join(DATA_DIR, f"vipa_{ts}.jpg"))

    def _dialog_save(self):
        fp = settings.save_dialog(
            initialdir=DATA_DIR,
            initialfile=f"vipa_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
            title="Save Image",
            filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")),
            defaultextension=".jpg",
        )
        if fp:
            self._save_frame(fp)

    def _switch_camera(self):
        if len(self.cameras) < 2:
            return
        self._cached_state[self.camera.serial] = dict(
            roi_model=self.roi_model,
            exp=self.camera.ExposureTime,
            gain=self.camera.Gain,
            auto=self.auto_exp,
            rect=self.rect_sensor,
            fit_rect=self.fit_rect_sensor,
            do_fitting=self.do_fitting,
            profiler_enabled=self.profiler_enabled,
            show_stats=self.show_stats,
        )
        self.processor = FrameProcessor()
        self.curr_idx = (self.curr_idx + 1) % len(self.cameras)
        self.camera = self.cameras[self.curr_idx]
        st = self._cached_state.get(self.camera.serial)
        if st:
            self.roi_model = st["roi_model"]
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = st["exp"]
            self.camera.Gain = st["gain"]
            self.auto_exp = st["auto"]
            self.rect_sensor = st["rect"]
            self.fit_rect_sensor = st["fit_rect"]
            self.do_fitting = st["do_fitting"]
            self.profiler_enabled = st["profiler_enabled"]
            self.show_stats = st["show_stats"]
        else:
            self.roi_model = ROIModel(self.camera.W, self.camera.H)
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = self.exposure_us
            self.camera.Gain = 0
            self.auto_exp = False
            self.rect_sensor = self.fit_rect_sensor = None
            self.do_fitting = self.show_stats = False
            self.profiler_enabled = False
        self.profiler_message = ''
        logging.info("Switched to %s", self.camera.model_name)

    def _toggle_web_server(self):
        if not self.web_server_enabled:
            if self._web_thread is None:
                from ..server import create_app

                app = create_app(self.session)
                self._web_thread = Thread(
                    target=lambda: app.run(
                        host="0.0.0.0", port=WEB_SERVER_PORT, debug=False, use_reloader=False
                    ),
                    daemon=True,
                )
                self._web_thread.start()
            self.web_server_enabled = True
            self.show_rect_stats = True
            logging.info(f"Web server started on http://localhost:{WEB_SERVER_PORT}")
        else:
            self.web_server_enabled = False
            self.show_rect_stats = False
            logging.info("Web server stopped and rect stats display disabled")

    def _sleep_for_fps(self, frame_dt: float):
        target = 1 / FPS_LIMIT
        if frame_dt < target:
            time.sleep(target - frame_dt)
            frame_dt = target
        self.avg_dt = 0.9 * self.avg_dt + 0.1 * frame_dt
