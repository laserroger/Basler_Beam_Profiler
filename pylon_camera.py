import cv2, sys, time, logging, os, tkinter as tk
import numpy as np
from tkinter import filedialog
from flask import Flask, jsonify, request, send_file
from threading import Thread
import json
from io import BytesIO

from pylon_camera_utils import ROIModel, classify_dots_grid
from basler import BaslerCamera
from blob_detector import blob_detector, render_blobs_with_img
from pypylon import pylon

# ------------------------------ CONFIG ---------------------------------- #


def app_dir():
    """Directory the app lives in: next to the .exe when frozen by PyInstaller,
    next to this script otherwise.  Everything the user is meant to see or edit
    (data/, camera_config.yaml, pylon_camera.json) lives here."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = app_dir()

MIN_EXPOSURE = 30  # µs
MAX_EXPOSURE = 100000  # µs
FPS_LIMIT = 60  # frames / s (UI update)
WINDOW_SIZE = 1100  # longest display edge in px
PAD_COLOR = (128, 128, 128)  # gray padding
DATA_DIR = os.path.join(APP_DIR, "data")
WEB_SERVER_PORT = 5000  # Web server port
STATIC_STATUS_JSON_PATH = os.path.join(APP_DIR, "pylon_camera.json")

# ------------------------------ LOGGING --------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------------ CAMERA ---------------------------------- #


factory = pylon.TlFactory.GetInstance()
available = factory.EnumerateDevices()
cameras = [BaslerCamera(mode="16Bit", device_idx=i) for i, _ in enumerate(available)]

if not cameras:
    logging.error("No Basler cameras detected – aborting.")
    sys.exit(1)

# cache per‑camera state so switching is seamless
cached_state: dict[str, dict] = {}

# --------------------------- GUI HELPERS ------------------------------- #


def choose_filename():
    root = tk.Tk()
    root.withdraw()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default = f"vipa_{timestamp}.jpg"
    fp = filedialog.asksaveasfilename(
        initialdir=DATA_DIR,
        initialfile=default,
        title="Save Image",
        filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")),
    )
    root.destroy()
    return fp


# --------------------------- APPLICATION ------------------------------- #


class BaslerViewer:
    def __init__(self):
        self.curr_idx = 0
        self.camera = cameras[self.curr_idx]
        self.roi_model = ROIModel(self.camera.W, self.camera.H)
        self.camera.ROI = self.roi_model.tuple
        self._auto_exp = False
        self.do_fitting = False
        self.show_stats = False  # toggled by 'g'
        self.row_col_fitting = False  # toggled by 'h'
        self.last_mouse = (0, 0)  # sensor coords
        self.rect_disp = None  # generic display rectangle (x1,y1,x2,y2)
        self.rect_sensor = None  # sensor coords
        self.fit_rect_disp = None  # fitting‑region rectangle (display coords)
        self.fit_rect_sensor = None  # fitting‑region rectangle (sensor coords)
        self.exposure_us = 200  # default, in us
        self.avg_dt = 1 / FPS_LIMIT
        self.avg_portion = 0.8
        self._init_window()
        self.stats_std_dx = 0
        self.stats_std_dy = 0
        self.stats_sigma_std = 0

        # Web server related
        self.current_frame = None
        self.latest_spots = []
        self.latest_stats = {}
        self.web_server_enabled = False  # toggled by 'w'
        self.show_rect_stats = False  # show real-time rect stats on screen
        self.web_app = None
        self.web_server_thread = None
        self.current_rect_stats = None  # store current rect stats for display

    @property
    def auto_exp(self):
        return self._auto_exp

    @auto_exp.setter
    def auto_exp(self, value):
        self._auto_exp = value
        if value:
            self.camera.AutoExposureOn(range=(MIN_EXPOSURE, MAX_EXPOSURE))
        else:
            self.camera.AutoExposureOff()
            self.exposure_us = int(
                np.clip(self.exposure_us, MIN_EXPOSURE, MAX_EXPOSURE)
            )

    # ---------------------- window & callbacks ---------------------- #
    def _init_window(self):
        cv2.namedWindow("Basler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Basler", WINDOW_SIZE, WINDOW_SIZE)
        cv2.setMouseCallback("Basler", self.on_mouse)

    def _create_web_app(self):
        """创建Flask Web应用"""
        app = Flask(__name__)

        @app.route("/api/status")
        def get_status():
            """获取相机基本状态"""
            return jsonify(
                {
                    "camera_model": self.camera.model_name,
                    "camera_serial": self.camera.serial,
                    "exposure_time": self.camera.ExposureTime,
                    "roi": self.camera.ROI,
                    "auto_exposure": self.auto_exp,
                    "fitting_enabled": self.do_fitting,
                    "rect_sensor": self.rect_sensor,
                    "fit_rect_sensor": self.fit_rect_sensor,
                }
            )

        @app.route("/api/rect_stats")
        def get_rect_stats():
            """获取矩形区域统计信息"""
            if self.current_frame is None:
                return jsonify({"error": "No frame available"}), 400

            if self.rect_sensor is None:
                return jsonify({"error": "No rectangle defined"}), 400

            # 获取矩形区域在ROI中的坐标
            w_roi, h_roi, ox, oy = self.camera.ROI
            sx1, sy1, sx2, sy2 = self.rect_sensor

            # 转换为ROI内的坐标
            rx1 = max(0, sx1 - ox)
            ry1 = max(0, sy1 - oy)
            rx2 = min(w_roi, sx2 - ox)
            ry2 = min(h_roi, sy2 - oy)

            if rx1 >= rx2 or ry1 >= ry2:
                return jsonify({"error": "Rectangle outside of ROI"}), 400

            # 提取矩形区域
            rect_region = self.current_frame[int(ry1) : int(ry2), int(rx1) : int(rx2)]

            # 计算统计信息
            stats = {
                "rectangle_bounds": {
                    "sensor_coords": [int(sx1), int(sy1), int(sx2), int(sy2)],
                    "roi_coords": [int(rx1), int(ry1), int(rx2), int(ry2)],
                    "width_pixels": int(rx2 - rx1),
                    "height_pixels": int(ry2 - ry1),
                    "width_um": float((rx2 - rx1) * self.camera.pixel_size * 1e6),
                    "height_um": float((ry2 - ry1) * self.camera.pixel_size * 1e6),
                },
                "pixel_stats": {
                    "sum": int(np.sum(rect_region)),
                    "mean": float(np.mean(rect_region)),
                    "std": float(np.std(rect_region)),
                    "min": int(np.min(rect_region)),
                    "max": int(np.max(rect_region)),
                    "median": float(np.median(rect_region)),
                    "total_pixels": int(rect_region.size),
                },
                "timestamp": time.time(),
            }

            return jsonify(stats)

        @app.route("/api/fit_rect_stats")
        def get_fit_rect_stats():
            """获取拟合矩形区域统计信息"""
            if self.current_frame is None:
                return jsonify({"error": "No frame available"}), 400

            if self.fit_rect_sensor is None:
                return jsonify({"error": "No fitting rectangle defined"}), 400

            # 获取拟合矩形区域在ROI中的坐标
            w_roi, h_roi, ox, oy = self.camera.ROI
            sx1, sy1, sx2, sy2 = self.fit_rect_sensor

            # 转换为ROI内的坐标
            rx1 = max(0, sx1 - ox)
            ry1 = max(0, sy1 - oy)
            rx2 = min(w_roi, sx2 - ox)
            ry2 = min(h_roi, sy2 - oy)

            if rx1 >= rx2 or ry1 >= ry2:
                return jsonify({"error": "Fitting rectangle outside of ROI"}), 400

            # 提取矩形区域
            rect_region = self.current_frame[int(ry1) : int(ry2), int(rx1) : int(rx2)]

            # 计算统计信息
            stats = {
                "rectangle_bounds": {
                    "sensor_coords": [int(sx1), int(sy1), int(sx2), int(sy2)],
                    "roi_coords": [int(rx1), int(ry1), int(rx2), int(ry2)],
                    "width_pixels": int(rx2 - rx1),
                    "height_pixels": int(ry2 - ry1),
                    "width_um": float((rx2 - rx1) * self.camera.pixel_size * 1e6),
                    "height_um": float((ry2 - ry1) * self.camera.pixel_size * 1e6),
                },
                "pixel_stats": {
                    "sum": int(np.sum(rect_region)),
                    "mean": float(np.mean(rect_region)),
                    "std": float(np.std(rect_region)),
                    "min": int(np.min(rect_region)),
                    "max": int(np.max(rect_region)),
                    "median": float(np.median(rect_region)),
                    "total_pixels": int(rect_region.size),
                },
                "timestamp": time.time(),
            }

            return jsonify(stats)

        @app.route("/api/spots")
        def get_spots():
            """获取检测到的光斑信息"""
            return jsonify(
                {
                    "spots": self.latest_spots,
                    "spot_count": len(self.latest_spots),
                    "stats": self.latest_stats,
                    "timestamp": time.time(),
                }
            )

        @app.route("/api/image")
        def get_image():
            """获取当前图像"""
            if self.current_frame is None:
                return jsonify({"error": "No frame available"}), 400

            # 转换为8位图像
            if self.camera.mode == "16Bit":
                img_8bit = cv2.convertScaleAbs(self.current_frame, alpha=1 / 256.0)
            else:
                img_8bit = self.current_frame

            # 编码为JPEG
            _, buffer = cv2.imencode(".jpg", img_8bit)

            return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")

        @app.route("/api/image_within_rect")
        def get_image_within_rect():
            """获取矩形区域内的当前图像"""
            if self.current_frame is None:
                return jsonify({"error": "No frame available"}), 400

            if self.rect_sensor is None:
                return jsonify({"error": "No rectangle defined"}), 400

            # 获取矩形区域在ROI中的坐标
            w_roi, h_roi, ox, oy = self.camera.ROI
            sx1, sy1, sx2, sy2 = self.rect_sensor

            # 转换为ROI内的坐标
            rx1 = max(0, sx1 - ox)
            ry1 = max(0, sy1 - oy)
            rx2 = min(w_roi, sx2 - ox)
            ry2 = min(h_roi, sy2 - oy)

            if rx1 >= rx2 or ry1 >= ry2:
                return jsonify({"error": "Rectangle outside of ROI"}), 400

            # 提取矩形区域
            rect_region = self.current_frame[int(ry1) : int(ry2), int(rx1) : int(rx2)]

            # 转换为8位图像
            if self.camera.mode == "16Bit":
                img_8bit = cv2.convertScaleAbs(rect_region, alpha=1 / 256.0)
            else:
                img_8bit = rect_region

            # 编码为JPEG
            _, buffer = cv2.imencode(".jpg", img_8bit)

            return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")

        @app.route("/api/set_rect", methods=["POST"])
        def set_rect():
            """设置矩形区域"""
            data = request.get_json()
            if not data or "coords" not in data:
                return jsonify({"error": "Invalid request data"}), 400

            coords = data["coords"]
            if len(coords) != 4:
                return jsonify({"error": "Coordinates must be [x1, y1, x2, y2]"}), 400

            self.rect_sensor = tuple(int(c) for c in coords)
            return jsonify(
                {"message": "Rectangle set successfully", "coords": self.rect_sensor}
            )

        @app.route("/api/set_fit_rect", methods=["POST"])
        def set_fit_rect():
            """设置拟合矩形区域"""
            data = request.get_json()
            if not data or "coords" not in data:
                return jsonify({"error": "Invalid request data"}), 400

            coords = data["coords"]
            if len(coords) != 4:
                return jsonify({"error": "Coordinates must be [x1, y1, x2, y2]"}), 400

            self.fit_rect_sensor = tuple(int(c) for c in coords)
            return jsonify(
                {
                    "message": "Fitting rectangle set successfully",
                    "coords": self.fit_rect_sensor,
                }
            )

        @app.route("/api/clear_rect", methods=["POST"])
        def clear_rect():
            """清除矩形区域"""
            self.rect_sensor = None
            return jsonify({"message": "Rectangle cleared"})

        @app.route("/api/clear_fit_rect", methods=["POST"])
        def clear_fit_rect():
            """清除拟合矩形区域"""
            self.fit_rect_sensor = None
            return jsonify({"message": "Fitting rectangle cleared"})

        @app.route("/")
        def index():
            """简单的Web界面"""
            html = """
            <!DOCTYPE html>

            <html>
            <head>
                <title>Basler Camera Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .endpoint { margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }
                    .method { color: #007acc; font-weight: bold; }
                    pre { background: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }
                </style>
            </head>
            <body>
                <h1>Basler Camera Web Server</h1>
                <h2>Available Endpoints:</h2>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/status</code>
                    <p>获取相机基本状态信息</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/rect_stats</code>
                    <p>获取矩形区域像素统计信息（需要先设置矩形）</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/fit_rect_stats</code>
                    <p>获取拟合矩形区域像素统计信息（需要先设置拟合矩形）</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/spots</code>
                    <p>获取检测到的光斑信息</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/api/image</code>
                    <p>获取当前相机图像（JPEG格式）</p>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/set_rect</code>
                    <p>设置矩形区域，POST JSON: {"coords": [x1, y1, x2, y2]}</p>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/set_fit_rect</code>
                    <p>设置拟合矩形区域，POST JSON: {"coords": [x1, y1, x2, y2]}</p>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/clear_rect</code>
                    <p>清除矩形区域</p>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span> <code>/api/clear_fit_rect</code>
                    <p>清除拟合矩形区域</p>
                </div>

                <h2>示例用法:</h2>
                <pre>
# 获取状态
curl http://localhost:5000/api/status

# 设置矩形区域（传感器坐标）
curl -X POST -H "Content-Type: application/json" \\
     -d '{"coords": [100, 100, 300, 300]}' \\
     http://localhost:5000/api/set_rect

# 获取矩形区域统计
curl http://localhost:5000/api/rect_stats

# 获取当前图像
curl http://localhost:5000/api/image -o current_image.jpg
                </pre>
            </body>
            </html>
            """
            return html

        return app

    def on_mouse(self, event, x, y, flags, _param):
        # calculate padding
        w_roi, h_roi, ox, oy = self.camera.ROI
        scale = WINDOW_SIZE / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        pad_l = (WINDOW_SIZE - view_w) // 2
        pad_t = (WINDOW_SIZE - view_h) // 2

        # check if mouse is in the image area
        if (pad_l <= x < pad_l + view_w) and (pad_t <= y < pad_t + view_h):
            sensor_x, sensor_y = self.roi_model.display_to_sensor(
                x, y, WINDOW_SIZE, pad_l, pad_t
            )
            self.last_mouse = (int(sensor_x), int(sensor_y))
        else:
            # mouse is outside the image area
            pass

        ctrl_down = flags & cv2.EVENT_FLAG_CTRLKEY

        # ---------------------------------------------------------- scroll wheel
        if event == cv2.EVENT_MOUSEWHEEL:
            if ctrl_down:
                # change aspect ratio while keeping long edge fixed
                if np.abs(self.roi_model.aspect) < 0.01:
                    return  # avoid division by zero
                factor = 0.95 if flags > 0 else 1.05
                new_aspect = self.roi_model.aspect * factor
                self.roi_model.keep_point_fixed(
                    self.last_mouse[0], self.last_mouse[1], new_aspect=new_aspect
                )
            else:
                # zoom in/out
                if np.abs(self.roi_model.scale) > 0.1:
                    delta = -0.05 if flags > 0 else 0.05
                else:
                    delta = -0.01 if flags > 0 else 0.01
                new_scale = self.roi_model.scale + delta
                self.roi_model.keep_point_fixed(
                    self.last_mouse[0], self.last_mouse[1], new_scale=new_scale
                )
            self.camera.ROI = self.roi_model.tuple

        # ---------------------------------------------------------- rectangle draw
        if event == cv2.EVENT_LBUTTONDOWN:
            if (pad_l <= x < pad_l + view_w) and (pad_t <= y < pad_t + view_h):
                if ctrl_down:
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
                x1, y1, x2, y2 = self.fit_rect_disp
                self.fit_rect_disp = None

                # convert the corners of the rectangle to sensor coordinates
                s_x1, s_y1 = self.roi_model.display_to_sensor(
                    min(x1, x2), min(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                s_x2, s_y2 = self.roi_model.display_to_sensor(
                    max(x1, x2), max(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                self.fit_rect_sensor = (int(s_x1), int(s_y1), int(s_x2), int(s_y2))
                self.save_status_to_json()

            elif self.rect_disp is not None:
                x1, y1, x2, y2 = self.rect_disp
                self.rect_disp = None

                # convert the corners of the rectangle to sensor coordinates
                s_x1, s_y1 = self.roi_model.display_to_sensor(
                    min(x1, x2), min(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                s_x2, s_y2 = self.roi_model.display_to_sensor(
                    max(x1, x2), max(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                self.rect_sensor = (int(s_x1), int(s_y1), int(s_x2), int(s_y2))
                self.save_status_to_json()

    def save_status_to_json(self):
        """将当前状态保存到JSON文件"""
        status = {
            "camera_model": self.camera.model_name,
            "camera_serial": self.camera.serial,
            "exposure_time": self.camera.ExposureTime,
            "roi": self.camera.ROI,
            "auto_exposure": self.auto_exp,
            "fitting_enabled": self.do_fitting,
            "rect_sensor": self.rect_sensor,
            "fit_rect_sensor": self.fit_rect_sensor,
        }
        try:
            with open(STATIC_STATUS_JSON_PATH, "w") as f:
                json.dump(status, f, indent=4)
            # print(f"Status saved to {STATIC_STATUS_JSON_PATH}")
        except Exception as e:
            print(f"Error saving status to {STATIC_STATUS_JSON_PATH}: {e}")

    # ----------------------------- main loop ---------------------------- #
    def run(self):
        while True:
            t0 = time.time()
            frame = self.camera.grab_image()
            if frame is None:
                continue
            if self.camera.mode == "16Bit":
                frame_disp = cv2.convertScaleAbs(frame, alpha=1 / 256.0)
            else:
                frame_disp = frame
            self.frame = frame_disp  # keep for saving raw images
            self.current_frame = frame  # 保存原始帧供Web服务器使用

            # 计算实时矩形统计（如果启用了显示）
            if self.show_rect_stats and self.rect_sensor is not None:
                self._calculate_rect_stats()

            # ---------------------------------------------------- Blob detection
            spots = []
            if self.do_fitting:
                all_spots = blob_detector(frame)
                # restrict to fitting_rect if provided
                if self.fit_rect_sensor is not None:
                    x1, y1, x2, y2 = self.fit_rect_sensor
                    for s in all_spots:
                        s_x, s_y = self.roi_model.coord_in_roi_to_sensor(s["x"], s["y"])
                        if x1 <= s_x <= x2 and y1 <= s_y <= y2:
                            spots.append(s)
                else:
                    spots = all_spots

            # 更新最新的光斑信息供Web服务器使用
            self.latest_spots = spots

            # create RGB overlay image
            rgb = cv2.cvtColor(frame_disp, cv2.COLOR_GRAY2RGB)
            if self.do_fitting and spots:
                # 常规拟合模式
                if not self.row_col_fitting:
                    rgb = render_blobs_with_img(
                        rgb,
                        spots,
                        rgb=True,
                        render_axis=True,
                        render_xy=True,
                        render_sigma=True,
                        pixel_size=self.camera.pixel_size,
                    )

            # ------------------- resize & pad to square -------------- #
            w_roi, h_roi, *_ = self.camera.ROI
            scale = WINDOW_SIZE / max(w_roi, h_roi)
            view = cv2.resize(rgb, (int(w_roi * scale), int(h_roi * scale)))
            pad_l = (WINDOW_SIZE - view.shape[1]) // 2
            pad_r = WINDOW_SIZE - view.shape[1] - pad_l
            pad_t = (WINDOW_SIZE - view.shape[0]) // 2
            pad_b = WINDOW_SIZE - view.shape[0] - pad_t
            disp = cv2.copyMakeBorder(
                view, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=PAD_COLOR
            )

            # ------------------------- overlays ---------------------- #
            next_y = self._draw_hud(disp)
            self._calc_spot_statistics(spots)  # update stats

            # 更新最新统计信息供Web服务器使用
            if self.web_server_enabled:
                self._update_web_stats()

            # 显示矩形统计信息（实时显示）
            next_y = self._draw_rect_stats(disp, next_y)

            self._draw_stats_bar(disp, next_y)  # if stats enabled draws
            self._draw_rectangles(disp, pad_l, pad_t)

            # 行列拟合可视化模式
            if self.row_col_fitting:
                self._draw_row_col_visualization(disp)

            cv2.imshow("Basler", disp)
            key = cv2.waitKeyEx(1)
            if not self._handle_key(key):
                break

            # fps regulation ---------------------------------------- #
            self._sleep_for_fps(time.time() - t0)

        self.camera.close()
        cv2.destroyAllWindows()

        # 停止Web服务器
        if self.web_server_thread and self.web_server_thread.is_alive():
            # Flask服务器会在主线程结束时自动停止
            pass

    def _start_web_server(self):
        """在后台线程启动Web服务器"""
        if self.web_server_enabled:
            return  # 已经启动了

        # 创建Web应用
        if self.web_app is None:
            self.web_app = self._create_web_app()

        def run_server():
            try:
                self.web_app.run(
                    host="0.0.0.0",
                    port=WEB_SERVER_PORT,
                    debug=False,
                    use_reloader=False,
                )
            except Exception as e:
                logging.error(f"Web server error: {e}")

        self.web_server_thread = Thread(target=run_server, daemon=True)
        self.web_server_thread.start()
        self.web_server_enabled = True

    def _stop_web_server(self):
        """停止Web服务器"""
        if not self.web_server_enabled:
            return

        self.web_server_enabled = False
        # Flask服务器会在主线程结束时自动停止
        # 由于使用了daemon线程，它会在主程序结束时自动终止

    def _update_web_stats(self):
        """更新最新统计信息供Web服务器使用"""
        pixel_to_um = self.camera.pixel_size * 1e6

        # 基本统计信息
        stats = {
            "basic_stats": {
                "dx_std": (
                    float(np.std(self.stats_dx)) if len(self.stats_dx) > 0 else 0.0
                ),
                "dy_std": (
                    float(np.std(self.stats_dy)) if len(self.stats_dy) > 0 else 0.0
                ),
                "dx_mean_um": (
                    float(np.mean(self.stats_dx) * pixel_to_um)
                    if len(self.stats_dx) > 0
                    else 0.0
                ),
                "dy_mean_um": (
                    float(np.mean(self.stats_dy) * pixel_to_um)
                    if len(self.stats_dy) > 0
                    else 0.0
                ),
                "sigma_std": float(self.stats_sigma_std),
            }
        }

        # 行列统计信息
        if hasattr(self, "grid_stats") and self.grid_stats:
            stats["grid_stats"] = self.grid_stats.copy()
            # 添加微米单位的统计
            if "rows" in stats["grid_stats"]:
                for key in ["avg_mean_dx", "avg_std_x", "avg_std_y"]:
                    if key in stats["grid_stats"]["rows"]:
                        stats["grid_stats"]["rows"][f"{key}_um"] = (
                            stats["grid_stats"]["rows"][key] * pixel_to_um
                        )

            if "columns" in stats["grid_stats"]:
                for key in ["avg_mean_dy", "avg_std_y", "avg_std_x"]:
                    if key in stats["grid_stats"]["columns"]:
                        stats["grid_stats"]["columns"][f"{key}_um"] = (
                            stats["grid_stats"]["columns"][key] * pixel_to_um
                        )

        # 行列数量
        if hasattr(self, "rows") and hasattr(self, "columns"):
            stats["row_col_counts"] = {
                "num_rows": len(self.rows),
                "num_columns": len(self.columns),
            }

        self.latest_stats = stats

    # ------------------------------------------------------------------ #
    # HUD & overlays
    # ------------------------------------------------------------------ #
    def _draw_hud(self, img):
        """Draw HUD on top of gray padding (#2). Returns the y coordinate after
        the last HUD line so stats bar can follow."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 26
        col = (255, 255, 255)
        HUD_X = 10
        HUD_Y = 5
        y = HUD_Y

        exposure_now = self.camera.ExposureTime
        if self.auto_exp:
            exposure_txt = f"Auto = {exposure_now:.0f}"
        else:
            exposure_txt = f"{exposure_now}"

        rect_w_um = (
            self.camera.pixel_size * (self.rect_sensor[2] - self.rect_sensor[0]) * 1e6
            if self.rect_sensor
            else 0
        )
        rect_h_um = (
            self.camera.pixel_size * (self.rect_sensor[3] - self.rect_sensor[1]) * 1e6
            if self.rect_sensor
            else 0
        )

        web_status = "ON" if self.web_server_enabled else "OFF"
        rect_stats_status = "ON" if self.show_rect_stats else "OFF"

        items = [
            f"Camera: {self.camera.model_name}, FPS: {1/self.avg_dt:4.1f}",
            f"ROI: {self.camera.ROI}  Mouse: {self.last_mouse}",
            f"Exposure: {exposure_txt} us",
            f"Rect: {self.rect_sensor}, W = {rect_w_um:.1f} um, H = {rect_h_um:.1f} um",
            f"Fitting: {self.do_fitting}, Stats: {self.show_stats}, Row-Col: {self.row_col_fitting}",
            f"Web Server: {web_status}, Rect Stats: {rect_stats_status} (Press 'w' to toggle)",
            # f"exposure_sync: {self.camera.exposure_line_enabled} (Press 'y' to toggle)",
            f"exposure_sync: {self.camera.userdefined_line_enabled} (Press 'y' to toggle)",
        ]
        for txt in items:
            y += line_height
            cv2.putText(img, txt, (HUD_X, y), font, 0.7, col, 2, cv2.LINE_AA)
        return y  # next y for further drawings

    def _draw_rectangles(self, img, pad_l: int = 0, pad_t: int = 0):
        # get ROI size and scale
        w_roi, h_roi, ox, oy = self.camera.ROI
        scale = WINDOW_SIZE / max(w_roi, h_roi)

        # draw the ROI rectangle (when dragging)
        if self.rect_disp is not None:
            x1, y1, x2, y2 = self.rect_disp
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if self.fit_rect_disp is not None:
            x1, y1, x2, y2 = self.fit_rect_disp
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw the ROI rectangle (when saved)
        if self.rect_sensor is not None:
            sx1, sy1, sx2, sy2 = self.rect_sensor
            p1_x, p1_y = self.roi_model.sensor_to_display(
                sx1, sy1, WINDOW_SIZE, pad_l, pad_t
            )
            p2_x, p2_y = self.roi_model.sensor_to_display(
                sx2, sy2, WINDOW_SIZE, pad_l, pad_t
            )
            cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (255, 255, 255), 2)

        if self.fit_rect_sensor is not None:
            sx1, sy1, sx2, sy2 = self.fit_rect_sensor
            p1_x, p1_y = self.roi_model.sensor_to_display(
                sx1, sy1, WINDOW_SIZE, pad_l, pad_t
            )
            p2_x, p2_y = self.roi_model.sensor_to_display(
                sx2, sy2, WINDOW_SIZE, pad_l, pad_t
            )
            cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), 2)

    def _calc_spot_statistics(self, spots):
        # 原始统计计算
        if len(spots) > 1:
            xs = np.array([s["x"] for s in spots])
            ys = np.array([s["y"] for s in spots])
            # argsort x
            xidx = np.argsort(xs)
            xs = xs[xidx]
            ys = ys[xidx]
            stats_dx = np.array([xs[i + 1] - xs[i] for i in range(len(xs) - 1)])
            stats_dy = np.array([ys[i + 1] - ys[i] for i in range(len(ys) - 1)])
            #
            sigma_0s = np.array([s["sigma_0"] for s in spots])
            sigma_1s = np.array([s["sigma_1"] for s in spots])
            stats_sigma = np.sqrt(sigma_0s * sigma_1s)

            # 将点分类为行和列
            rows, columns, grid_stats = classify_dots_grid(spots, eps=20)
        else:
            stats_dx = np.array([0.0])
            stats_dy = np.array([0.0])
            stats_sigma = np.array([0.0])
            rows, columns, grid_stats = [], [], {}

        self.stats_dx = stats_dx
        self.stats_dy = stats_dy
        self.stats_sigma = stats_sigma
        self.grid_stats = grid_stats
        self.rows = rows
        self.columns = columns

    def _calculate_rect_stats(self):
        """计算矩形区域的像素统计信息"""
        if self.current_frame is None or self.rect_sensor is None:
            self.current_rect_stats = None
            return

        # 获取矩形区域在ROI中的坐标
        w_roi, h_roi, ox, oy = self.camera.ROI
        sx1, sy1, sx2, sy2 = self.rect_sensor

        # 转换为ROI内的坐标
        rx1 = max(0, sx1 - ox)
        ry1 = max(0, sy1 - oy)
        rx2 = min(w_roi, sx2 - ox)
        ry2 = min(h_roi, sy2 - oy)

        if rx1 >= rx2 or ry1 >= ry2:
            self.current_rect_stats = None
            return

        # 提取矩形区域
        rect_region = self.current_frame[int(ry1) : int(ry2), int(rx1) : int(rx2)]

        # 计算统计信息
        self.current_rect_stats = {
            "sum": int(np.sum(rect_region)),
            "mean": float(np.mean(rect_region)),
            "std": float(np.std(rect_region)),
            "min": int(np.min(rect_region)),
            "max": int(np.max(rect_region)),
            "median": float(np.median(rect_region)),
            "total_pixels": int(rect_region.size),
            "width_pixels": int(rx2 - rx1),
            "height_pixels": int(ry2 - ry1),
            "width_um": float((rx2 - rx1) * self.camera.pixel_size * 1e6),
            "height_um": float((ry2 - ry1) * self.camera.pixel_size * 1e6),
        }

    def _draw_rect_stats(self, display_img, start_y):
        """在屏幕上显示矩形区域统计信息"""
        if not self.show_rect_stats or self.current_rect_stats is None:
            return start_y

        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 24
        col = (0, 255, 255)  # 黄色
        text_x = 10

        # 显示矩形统计标题
        start_y += line_height
        cv2.putText(
            display_img,
            "=== Rect Stats ===",
            (text_x, start_y),
            font,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        stats = self.current_rect_stats
        items = [
            f"Sum: {stats['sum']:,}",
            f"Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}",
            f"Min: {stats['min']}, Max: {stats['max']}, Median: {stats['median']:.1f}",
            f"Size: {stats['width_pixels']}x{stats['height_pixels']} px ({stats['width_um']:.1f}x{stats['height_um']:.1f} um)",
            f"Total pixels: {stats['total_pixels']:,}",
        ]

        for txt in items:
            start_y += line_height
            cv2.putText(
                display_img, txt, (text_x, start_y), font, 0.6, col, 2, cv2.LINE_AA
            )

        return start_y + line_height  # 返回下一个可用的y位置

    def _draw_stats_bar(self, display_img, start_y):
        if not (self.show_stats):
            return

        pixel_to_um = self.camera.pixel_size * 1e6

        stats_dx = self.stats_dx
        stats_dy = self.stats_dy

        stats_std_dx = np.std(stats_dx)
        stats_std_dy = np.std(stats_dy)
        stats_dx_mean_um = np.mean(stats_dx) * self.camera.pixel_size * 1e6
        stats_dy_mean_um = np.mean(stats_dy) * self.camera.pixel_size * 1e6
        # calculate its second order statistics (curvature)
        # distinguishing plus and minus curvature
        if len(stats_dx) > 1:
            stats_dx_curvature = np.mean(np.diff(stats_dx, 1))
        else:
            stats_dx_curvature = 0.0
        if len(stats_dy) > 1:
            stats_dy_curvature = np.mean(np.diff(stats_dy, 1))
        else:
            stats_dy_curvature = 0.0

        if stats_std_dx is not None:
            bar_factor = 5  # scaling factor (adjust as needed)
            max_bar_length = 100  # maximum bar length for display
            line_height = 26
            bar_y = start_y + line_height  # start after HUD
            text_x = 10
            self.stats_std_dx = (
                self.avg_portion * self.stats_std_dx
                + (1 - self.avg_portion) * stats_std_dx
            )
            self.stats_std_dy = (
                self.avg_portion * self.stats_std_dy
                + (1 - self.avg_portion) * stats_std_dy
            )
            stats_std_dx = self.stats_std_dx
            stats_std_dy = self.stats_std_dy

            # 原始统计显示代码
            cv2.putText(
                display_img,
                f"std(x): {stats_std_dx:.2f}",
                (text_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            bar_x = text_x + 160
            bar_length = int(min(stats_std_dx * bar_factor, max_bar_length))
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + bar_length, bar_y + 20),
                (0, 255, 255),
                -1,
            )
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + max_bar_length, bar_y + 20),
                (255, 255, 255),
                2,
            )

            text_x = bar_x + max_bar_length + 50
            cv2.putText(
                display_img,
                f"std(y): {stats_std_dy:.2f}",
                (text_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            bar_x = text_x + 160
            bar_length = int(min(stats_std_dy * bar_factor, max_bar_length))
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + bar_length, bar_y + 20),
                (0, 255, 255),
                -1,
            )
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + max_bar_length, bar_y + 20),
                (255, 255, 255),
                2,
            )
            # 显示平均dx和dy值，以微米为单位
            text_mean_dx_x = bar_x + max_bar_length + 50
            cv2.putText(
                display_img,
                f"Dx: {stats_dx_mean_um:.1f} um",
                (text_mean_dx_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            text_mean_dy_x = text_mean_dx_x + 200
            cv2.putText(
                display_img,
                f"Dy: {stats_dy_mean_um:.1f} um",
                (text_mean_dy_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if self.show_stats and not self.row_col_fitting:
                # display the curvature values in the next line, similar to std(x) and std(y)
                # the bar factor here is 2.5, bar display from -20 to 20
                bar_factor = 2.5  # scaling factor for curvature
                max_bar_length = 100  # maximum bar length for curvature display
                half_bar_length = int(max_bar_length / 2)
                next_line_y = bar_y + 30
                text_x = 10
                cv2.putText(
                    display_img,
                    f"CurX: {stats_dx_curvature:.2f}",
                    (text_x, next_line_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                bar_x = text_x + 160
                bar_length = int(
                    min(abs(stats_dx_curvature) * bar_factor, half_bar_length)
                )

                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + max_bar_length, next_line_y + 20),
                    (255, 255, 255),
                    2,
                )
                if stats_dx_curvature < 0:  # from left to center
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length - bar_length, next_line_y),
                        (bar_x + half_bar_length, next_line_y + 20),
                        (0, 0, 255),  # red for negative curvature
                        -1,
                    )
                else:
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length, next_line_y),
                        (bar_x + half_bar_length + bar_length, next_line_y + 20),
                        (0, 255, 0),  # green for positive curvature
                        -1,
                    )
                text_x = bar_x + max_bar_length + 50
                cv2.putText(
                    display_img,
                    f"CurY: {stats_dy_curvature:.2f}",
                    (text_x, next_line_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                bar_x = text_x + 160
                bar_length = int(
                    min(abs(stats_dy_curvature) * bar_factor, half_bar_length)
                )
                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + max_bar_length, next_line_y + 20),
                    (255, 255, 255),
                    2,
                )
                if stats_dy_curvature < 0:  # from top to center
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length - bar_length, next_line_y),
                        (bar_x + half_bar_length, next_line_y + 20),
                        (0, 0, 255),  # red for negative curvature
                        -1,
                    )
                else:
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length, next_line_y),
                        (bar_x + half_bar_length + bar_length, next_line_y + 20),
                        (0, 255, 0),  # green for positive curvature
                        -1,
                    )

                # bar for sigma
                stats_sigma = self.stats_sigma
                # stats_sigma_std = np.std(stats_sigma)
                stats_sigma_std = self.avg_portion * self.stats_sigma_std + (
                    1 - self.avg_portion
                ) * np.std(stats_sigma)
                self.stats_sigma_std = stats_sigma_std  # update the instance variable
                bar_x = text_x + 320
                bar_factor = 20  # scaling factor for curvature
                max_bar_length = 100  # maximum bar length for curvature display
                cv2.putText(
                    display_img,
                    f"std(s): {stats_sigma_std:.3f}",
                    (bar_x, next_line_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                bar_x += 160
                bar_length = int(min(stats_sigma_std * bar_factor, max_bar_length))
                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + bar_length, next_line_y + 20),
                    (0, 255, 255),  # yellow for sigma std
                    -1,
                )
                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + max_bar_length, next_line_y + 20),
                    (255, 255, 255),  # white border
                    2,
                )
                # 显示平均sigma值，以像素为单位

        # ---- Row and column statistics ----

        grid_stats = self.grid_stats
        rows = self.rows
        columns = self.columns

        if self.row_col_fitting and grid_stats:
            for key in ["avg_mean_dx", "avg_std_x", "avg_std_y"]:
                if key in grid_stats["rows"]:
                    grid_stats["rows"][f"{key}_um"] = (
                        grid_stats["rows"][key] * pixel_to_um
                    )

            for key in ["avg_mean_dy", "avg_std_y", "avg_std_x"]:
                if key in grid_stats["columns"]:
                    grid_stats["columns"][f"{key}_um"] = (
                        grid_stats["columns"][key] * pixel_to_um
                    )

            # Row and column statistics
            next_line_y = bar_y + 40

            # Row statistics
            cv2.putText(
                display_img,
                f"NRow: {len(rows)}",
                (10, next_line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if "avg_mean_dx_um" in grid_stats["rows"]:
                cv2.putText(
                    display_img,
                    f"Dx: {grid_stats['rows']['avg_mean_dx_um']:.1f} um, std(x): {grid_stats['rows']['avg_std_x_um']:.1f} um, std(y): {grid_stats['rows']['avg_std_y_um']:.1f} um",
                    (200, next_line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 列统计
            next_line_y += 30
            cv2.putText(
                display_img,
                f"NCol: {len(columns)}",
                (10, next_line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if "avg_mean_dy_um" in grid_stats["columns"]:
                cv2.putText(
                    display_img,
                    f"Dy: {grid_stats['columns']['avg_mean_dy_um']:.1f} um, std(y): {grid_stats['columns']['avg_std_y_um']:.1f} um, std(x): {grid_stats['columns']['avg_std_x_um']:.1f} um",
                    (200, next_line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    def _draw_row_col_visualization(self, display_img):
        """使用连续色调(hue)绘制行列可视化，并绘制行列平均位置线"""
        if not (self.do_fitting and self.row_col_fitting):
            return

        # 将点分类为行和列
        rows = self.rows
        columns = self.columns

        # 获取ROI和填充信息用于坐标转换
        w_roi, h_roi, ox, oy = self.camera.ROI
        scale = WINDOW_SIZE / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        pad_l = (WINDOW_SIZE - view_w) // 2
        pad_t = (WINDOW_SIZE - view_h) // 2

        # 绘制行（水平线）
        for i, row in enumerate(rows):
            if len(row) < 2:
                continue

            # 生成连续色调的颜色 - 对行使用高饱和度
            hue = (i * 30) % 180  # 每行色调差30度，限制在0-180之间避免红色重复
            # 转换HSV到BGR (OpenCV使用BGR)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][
                0
            ].tolist()

            # 计算该行所有点的平均y值
            mean_y = np.mean([s["y"] for s in row])

            # 找出该行中x坐标的最小值和最大值
            min_x = min([s["x"] for s in row])
            max_x = max([s["x"] for s in row])

            # 转换为显示坐标
            s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(min_x, mean_y)
            s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(max_x, mean_y)

            d_x1, d_y1 = self.roi_model.sensor_to_display(
                s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            )
            d_x2, d_y2 = self.roi_model.sensor_to_display(
                s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            )

            # 绘制代表该行平均y值的水平实线
            cv2.line(display_img, (d_x1, d_y1), (d_x2, d_y2), color, 2, cv2.LINE_AA)

            # 按x坐标排序
            sorted_row = sorted(row, key=lambda s: s["x"])

            # # 连接每一行的点（用细线）
            # for j in range(len(sorted_row) - 1):
            #     p1 = sorted_row[j]
            #     p2 = sorted_row[j + 1]

            #     # 转换为显示坐标
            #     s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(p1["x"], p1["y"])
            #     s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(p2["x"], p2["y"])

            #     d_x1, d_y1 = self.roi_model.sensor_to_display(
            #         s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            #     )
            #     d_x2, d_y2 = self.roi_model.sensor_to_display(
            #         s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            #     )

            #     # 绘制细线连接点
            #     cv2.line(display_img, (d_x1, d_y1), (d_x2, d_y2), color, 1, cv2.LINE_AA)

            # 绘制圆圈标记每个点
            for p in sorted_row:
                s_x, s_y = self.roi_model.coord_in_roi_to_sensor(p["x"], p["y"])
                d_x, d_y = self.roi_model.sensor_to_display(
                    s_x, s_y, WINDOW_SIZE, pad_l, pad_t
                )
                cv2.circle(display_img, (d_x, d_y), 5, color, 2)

        # 绘制列（垂直虚线）
        for i, col in enumerate(columns):
            if len(col) < 2:
                continue

            # 生成连续色调的颜色 - 对列使用中等饱和度和明度
            hue = (i * 30) % 180  # 每列色调差30度
            # 转换HSV到BGR (OpenCV使用BGR)
            color = cv2.cvtColor(np.uint8([[[hue, 180, 255]]]), cv2.COLOR_HSV2BGR)[0][
                0
            ].tolist()

            # 计算该列所有点的平均x值
            mean_x = np.mean([s["x"] for s in col])

            # 找出该列中y坐标的最小值和最大值
            min_y = min([s["y"] for s in col])
            max_y = max([s["y"] for s in col])

            # 转换为显示坐标
            s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(mean_x, min_y)
            s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(mean_x, max_y)

            d_x1, d_y1 = self.roi_model.sensor_to_display(
                s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            )
            d_x2, d_y2 = self.roi_model.sensor_to_display(
                s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            )

            # 绘制代表该列平均x值的垂直虚线
            dash_length = 8  # 虚线长度
            gap_length = 4  # 虚线间隔

            # 计算两点间的总距离和方向向量
            dist = np.sqrt((d_x2 - d_x1) ** 2 + (d_y2 - d_y1) ** 2)
            if dist == 0:
                continue

            dx, dy = (d_x2 - d_x1) / dist, (d_y2 - d_y1) / dist

            # 绘制虚线
            pos = 0
            while pos < dist:
                # 虚线段起点
                start_x = int(d_x1 + dx * pos)
                start_y = int(d_y1 + dy * pos)

                # 虚线段终点
                end_pos = min(pos + dash_length, dist)
                end_x = int(d_x1 + dx * end_pos)
                end_y = int(d_y1 + dy * end_pos)

                cv2.line(
                    display_img,
                    (start_x, start_y),
                    (end_x, end_y),
                    color,
                    2,
                    cv2.LINE_AA,
                )

                # 移动到下一段虚线的起点
                pos = end_pos + gap_length

            # 按y坐标排序
            sorted_col = sorted(col, key=lambda s: s["y"])

            # # 连接每一列的点（用细线）
            # for j in range(len(sorted_col) - 1):
            #     p1 = sorted_col[j]
            #     p2 = sorted_col[j + 1]

            #     # 转换为显示坐标
            #     s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(p1["x"], p1["y"])
            #     s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(p2["x"], p2["y"])

            #     d_x1, d_y1 = self.roi_model.sensor_to_display(
            #         s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            #     )
            #     d_x2, d_y2 = self.roi_model.sensor_to_display(
            #         s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            #     )

            #     # 绘制细线连接点
            #     cv2.line(display_img, (d_x1, d_y1), (d_x2, d_y2), color, 1, cv2.LINE_AA)

            # 使用方形标记标识列中的每个点
            for p in sorted_col:
                s_x, s_y = self.roi_model.coord_in_roi_to_sensor(p["x"], p["y"])
                d_x, d_y = self.roi_model.sensor_to_display(
                    s_x, s_y, WINDOW_SIZE, pad_l, pad_t
                )
                cv2.drawMarker(display_img, (d_x, d_y), color, cv2.MARKER_SQUARE, 10, 2)

    # ------------------------------------------------------------------ #
    # key handling & helpers
    # ------------------------------------------------------------------ #
    def _handle_key(self, key: int) -> bool:
        if key == 27:  # ESC
            return False
        elif key in (2555904, 2424832, 2490368, 2621440):  # arrows
            self._adjust_exposure(key)
        elif key == ord("a"):
            self.auto_exp = not self.auto_exp
        elif key == ord("f"):
            self.do_fitting = not self.do_fitting
        elif key == ord("g"):
            self.show_stats = not self.show_stats  # toggle stats bar
        elif key == ord("h"):
            self.row_col_fitting = not self.row_col_fitting  # 切换行列拟合模式
        elif key == ord("c"):
            self.rect_sensor = None
        elif key == ord("v"):
            self.fit_rect_sensor = None  # clear fitting rect (#5)
        elif key == ord("s"):
            self._quick_save()
        elif key == ord("d"):
            self._dialog_save()
        elif key == ord("t"):
            self._switch_camera()
        elif key == ord("w"):
            # 切换Web服务器和实时矩形统计显示
            if not self.web_server_enabled:
                self._start_web_server()
                self.show_rect_stats = True
                logging.info(
                    f"Web server started on http://localhost:{WEB_SERVER_PORT}"
                )
                logging.info("Real-time rectangle stats display enabled")
            else:
                self._stop_web_server()
                self.show_rect_stats = False
                logging.info("Web server stopped and rect stats display disabled")
        elif key == ord("y"):
            # self.camera.set_exposure_line(not self.camera.exposure_line_enabled)
            self.camera.set_userdefined_line(not self.camera.userdefined_line_enabled)
            logging.info(
                f"userdefined line sync mode set to {self.camera.userdefined_line_enabled}"
            )
        #
        self.save_status_to_json()
        return True

    def _adjust_exposure(self, key):
        if self.auto_exp:
            return
        factor = {2555904: 1.1, 2424832: 0.9, 2490368: 10, 2621440: 0.1}[key]
        self.exposure_us = int(
            np.clip(self.exposure_us * factor, MIN_EXPOSURE, MAX_EXPOSURE)
        )
        self.camera.ExposureTime = self.exposure_us

    def _quick_save(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        jpg = os.path.join(DATA_DIR, f"vipa_{ts}.jpg")
        npy = os.path.join(DATA_DIR, f"vipa_{ts}.npy")
        frame = self.frame
        cv2.imwrite(jpg, frame)
        np.save(npy, frame)
        logging.info("Saved %s, %s", jpg, npy)

    def _dialog_save(self):
        fp = choose_filename()
        if not fp:
            return
        frame = self.frame
        cv2.imwrite(fp, frame)
        np.save(os.path.splitext(fp)[0] + ".npy", frame)
        logging.info("Saved %s (+ .npy)", fp)

    def _switch_camera(self):
        # save current
        cached_state[self.camera.serial] = dict(
            ROI_model=self.roi_model,
            exp=self.camera.ExposureTime,
            gain=self.camera.Gain,
            auto=self.auto_exp,
            rect=self.rect_sensor,
            fit_rect=self.fit_rect_sensor,
            do_fitting=self.do_fitting,
            show_stats=self.show_stats,
        )
        # next camera
        self.curr_idx = (self.curr_idx + 1) % len(cameras)
        self.camera = cameras[self.curr_idx]
        st = cached_state.get(self.camera.serial)
        if st:
            # load settings
            self.roi_model = st["ROI_model"]
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = st["exp"]
            self.camera.Gain = st["gain"]
            self.auto_exp = st["auto"]
            self.rect_sensor = st["rect"]
            self.fit_rect_sensor = st["fit_rect"]
            self.do_fitting = st["do_fitting"]
            self.show_stats = st["show_stats"]
        else:
            # reset settings
            self.roi_model = ROIModel(self.camera.W, self.camera.H)
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = self.exposure_us
            self.camera.Gain = 0
            self.auto_exp = False
            self.rect_sensor = None
            self.fit_rect_sensor = None
            self.do_fitting = False
            self.show_stats = False
        logging.info("Switched to %s", self.camera.model_name)

    def _sleep_for_fps(self, frame_dt):
        target_dt = 1 / FPS_LIMIT
        if frame_dt < target_dt:
            time.sleep(target_dt - frame_dt)
            frame_dt = target_dt
        self.avg_dt = 0.9 * self.avg_dt + 0.1 * frame_dt


# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    BaslerViewer().run()
