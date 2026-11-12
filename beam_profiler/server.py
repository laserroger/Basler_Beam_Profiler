"""HTTP API exposing camera status, frames and rectangle statistics.

Endpoints are documented in docs/web_api.md and listed on the index page."""

from __future__ import annotations

import time
from dataclasses import asdict
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template, g

from . import fitconfig
from .processing import crop_rect, region_stats
from .session import LiveSession
from .calibration import display_gray

ENDPOINTS = [
    ("GET", "/sizes", "Live D4sigma major/minor diameter plots and camera preview."),
    ("GET", "/api/spot_sizes", "D4sigma diameters in um and selected individual spot history."),
    ("POST", "/api/spot_sizes/select", "Toggle a spot by sensor coordinates; null clears all traces."),
    ("GET", "/api/spot_sizes/plot", "Live diameter charts as PNG."),
    ("GET", "/angular", "Single-shot angular-response plot (when enabled)."),
    ("GET", "/api/angular", "Latest single-frame angular response and measured timings."),
    ("GET", "/api/status", "Camera status (model, exposure, ROI, rectangles)."),
    ("GET", "/api/rect_stats", "Pixel statistics of the white rectangle."),
    ("GET", "/api/fit_rect_stats", "Pixel statistics of the green fitting rectangle."),
    ("GET", "/api/spots", "Detected spots and their statistics."),
    ("GET", "/api/fit_config", "Current spot-fitting parameters."),
    ("PUT", "/api/fit_config", "Update spot-fitting parameters (JSON body)."),
    ("GET", "/api/image", "Current frame as JPEG."),
    ('GET', '/api/raw_frame', 'Uncorrected sensor frame as .npy, with snapshot metadata headers.'),
    ('GET', '/api/calibration', 'Active flat-field correction state.'),
    ("GET", "/api/image_within_rect", "Current frame cropped to the white rectangle, as JPEG."),
    ("POST", "/api/set_rect", 'Set the white rectangle. JSON: {"coords": [x1, y1, x2, y2]}.'),
    ("POST", "/api/set_fit_rect", 'Set the fitting rectangle. JSON: {"coords": [x1, y1, x2, y2]}.'),
    ("POST", "/api/clear_rect", "Clear the white rectangle."),
    ("POST", "/api/clear_fit_rect", "Clear the fitting rectangle."),
]


def create_app(session: LiveSession) -> Flask:
    app = Flask(__name__)

    @app.before_request
    def read_snapshot():
        g.snapshot = session.read()
        g.result = g.snapshot.result


    @app.route('/sizes')
    def sizes_page():
        return render_template('sizes.html')

    def size_data():
        if g.snapshot.sizes is not None:
            return g.snapshot.sizes
        if g.result is not None:
            from .processing.sizes import spot_sizes
            return spot_sizes(g.result)
        return None

    @app.route('/api/spot_sizes')
    def get_sizes():
        data = size_data()
        if data is None:
            return jsonify(error='Waiting for camera frame'), 503
        response = jsonify(data)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.route('/api/spot_sizes/select', methods=['POST'])
    def select_size_spot():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Expected {"position": [sensor_x, sensor_y]}'), 400
        position = body.get('position', 'missing')
        if position is not None:
            if not isinstance(position, list) or len(position) != 2:
                return jsonify(error='position must be [sensor_x, sensor_y] or null'), 400
            try:
                position = [float(v) for v in position]
                if not np.isfinite(position).all():
                    raise ValueError()
            except (TypeError, ValueError, OverflowError):
                return jsonify(error='position must contain finite coordinates'), 400
            data = size_data()
            if data is None or (not data['spot_count'] and not data.get('traces')):
                return jsonify(error='No detected spots to select'), 409
            distances = np.hypot(np.asarray(data['x_sensor_px'], dtype=float)-position[0],
                                 np.asarray(data['y_sensor_px'], dtype=float)-position[1])
            finite = distances[np.isfinite(distances)]
            selected_nearby = any(np.hypot(t['x_sensor_px']-position[0], t['y_sensor_px']-position[1]) <= t['gate_px']
                                  for t in data.get('traces', []))
            if (not len(finite) or finite.min() > 40) and not selected_nearby:
                return jsonify(error='Click within 40 sensor pixels of a detected spot'), 400
        session.select_spot(position)
        return jsonify(queued=True), 202

    @app.route('/api/spot_sizes/reset_time', methods=['POST'])
    def reset_size_time():
        session.reset_trace_time()
        return jsonify(queued=True), 202

    @app.route('/api/spot_sizes/window', methods=['POST'])
    def set_size_window():
        body = request.get_json(silent=True)
        try:
            value = body['seconds']
            if isinstance(value, bool):
                raise ValueError()
            seconds = float(value)
            if not np.isfinite(seconds) or not 1 <= seconds <= 600:
                raise ValueError()
        except (TypeError, KeyError, ValueError, OverflowError):
            return jsonify(error='Time window must be between 1 and 600 seconds'), 400
        session.set_trace_window(seconds)
        return jsonify(queued=True), 202

    @app.route('/api/spot_sizes/plot')
    def get_size_plot():
        from .ui.sizes import render_sizes, parse_y_limits
        limits = None
        if 'y_min' in request.args or 'y_max' in request.args:
            try:
                limits = parse_y_limits(request.args.get('y_min'), request.args.get('y_max'))
            except (TypeError, ValueError, OverflowError):
                return jsonify(error='Use finite values with 0 <= Y min < Y max'), 400
        _, buffer = cv2.imencode('.png', render_sizes(size_data(), y_limits=limits))
        response = send_file(BytesIO(buffer.tobytes()), mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.route('/angular')
    def angular_page():
        return render_template('angular.html')

    @app.route('/api/angular')
    def angular_data():
        if g.snapshot.angular is None:
            return jsonify(valid=False, error='Angular monitor is not enabled'), 409
        return jsonify(g.snapshot.angular)

    @app.route('/api/preview')
    def camera_preview():
        if g.result is None:
            return jsonify(error='Waiting for camera frame'), 503
        frame, roi, rect = g.result.frame, g.result.roi, g.result.options.fit_rect
        gray = display_gray(frame, g.result.full_scale)
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if rect is not None:
            _, _, ox, oy = roi
            x1, y1, x2, y2 = rect
            cv2.rectangle(preview, (int(x1-ox), int(y1-oy)),
                          (int(x2-ox-1), int(y2-oy-1)), (0,255,0), 1)
        data = g.snapshot.sizes
        if data is not None:
            for index, (x, y) in enumerate(zip(data['x_sensor_px'], data['y_sensor_px']), 1):
                if x is None or y is None:
                    continue
                point = (round(x-roi[2]), round(y-roi[3]))
                cv2.circle(preview, point, 22, (155,155,155), 1, cv2.LINE_AA)
                label = f'{index} ({x:.0f},{y:.0f})'
                label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0]
                label_x = point[0]+25 if point[0]+25+label_width < preview.shape[1] else point[0]-25-label_width
                cv2.putText(preview, label, (max(0,label_x),point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (230,230,230), 1, cv2.LINE_AA)
            from .ui.sizes import bgr
            for selected in data.get('traces', []):
                point = (round(selected['x_sensor_px']-roi[2]), round(selected['y_sensor_px']-roi[3]))
                color = bgr(selected['color'])
                cv2.circle(preview, point, 28, color, 2, cv2.LINE_AA)
                cv2.putText(preview, f"Trace #{selected['id']}", (max(0,point[0]-30),point[1]-36),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', preview)
        response = send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store'
        return response

    def rect_stats_response(rect, label):
        if g.result is None:
            return jsonify({"error": "No frame available"}), 400
        if rect is None:
            return jsonify({"error": f"No {label} defined"}), 400
        stats = region_stats(
            g.result.frame, g.result.roi, rect, g.result.pixel_size
        )
        if stats is None:
            return jsonify({"error": f"{label} outside of ROI"}), 400
        bounds_keys = (
            "sensor_coords", "roi_coords",
            "width_pixels", "height_pixels", "width_um", "height_um",
        )
        return jsonify(
            {
                "rectangle_bounds": {k: stats[k] for k in bounds_keys},
                "pixel_stats": {k: v for k, v in stats.items() if k not in bounds_keys},
                "timestamp": g.result.timestamp,
            }
        )

    def jpeg_response(img):
        img = display_gray(img, g.result.full_scale)
        _, buffer = cv2.imencode(".jpg", img)
        return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")

    def parse_coords():
        data = request.get_json()
        if not isinstance(data, dict) or "coords" not in data:
            return None, (jsonify({"error": "Invalid request data"}), 400)
        coords = data["coords"]
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            return None, (jsonify({"error": "Coordinates must be [x1, y1, x2, y2]"}), 400)
        try:
            return tuple(int(c) for c in coords), None
        except (TypeError, ValueError, OverflowError):
            return None, (jsonify({"error": "Coordinates must be finite numbers"}), 400)

    @app.route("/api/status")
    def get_status():
        return jsonify(g.snapshot.status)

    @app.route('/api/calibration')
    def get_calibration():
        return jsonify(g.snapshot.status.get('calibration', dict(enabled=False)))

    @app.route('/api/raw_frame')
    def get_raw_frame():
        if g.result is None:
            return jsonify(error='No frame available'), 400
        raw = g.result.raw_frame if g.result.raw_frame is not None else g.result.frame
        buffer = BytesIO()
        np.save(buffer, raw, allow_pickle=False)
        buffer.seek(0)
        response = send_file(buffer, mimetype='application/octet-stream', download_name='raw_frame.npy')
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Frame-Timestamp'] = str(g.result.timestamp)
        response.headers['X-Camera-Serial'] = str(g.snapshot.status.get('camera_serial', ''))
        response.headers['X-Exposure-Us'] = str(g.snapshot.status.get('exposure_time', ''))
        response.headers['X-Gain-Db'] = str(g.snapshot.status.get('gain_db', ''))
        response.headers['X-Pixel-Format'] = str(g.snapshot.status.get('pixel_format', ''))
        response.headers['X-ADC-Bit-Depth'] = str(g.snapshot.status.get('adc_bit_depth') or '')
        return response

    @app.route("/api/rect_stats")
    def get_rect_stats():
        return rect_stats_response(g.result.options.rect if g.result else None, "rectangle")

    @app.route("/api/fit_rect_stats")
    def get_fit_rect_stats():
        return rect_stats_response(g.result.options.fit_rect if g.result else None, "fitting rectangle")

    @app.route("/api/spots")
    def get_spots():
        if g.result is None:
            return jsonify(spots=[], spot_count=0, stats={}, timestamp=time.time())
        spots = g.result.spots
        return jsonify(
            {
                "spots": [dict(s, diameter_major_um=2*s["sigma_0"]*g.result.pixel_size*1e6,
                               diameter_minor_um=2*s["sigma_1"]*g.result.pixel_size*1e6)
                          for s in spots.to_json()],
                "size_convention": "D4sigma",
                "spot_count": len(spots),
                "stats": g.result.statistics.summary(g.result.pixel_size),
                "saturated": g.result.saturated.tolist(),
                "profiler_message": g.result.profiler_message,
                "timestamp": g.result.timestamp,
            }
        )

    @app.route("/api/fit_config", methods=["GET", "PUT"])
    def fit_config():
        config = fitconfig.active()
        if request.method == "PUT":
            body = request.get_json(silent=True) or {}
            if not isinstance(body, dict):
                return jsonify(error="Expected a settings object"), 400
            unknown = set(body) - {s.name for s in fitconfig.SETTINGS}
            if unknown:
                return jsonify({"error": f"unknown settings: {sorted(unknown)}"}), 400
            config = config.replace(**body)
            session.set_fit_config(body)
        return jsonify(
            {
                "config": asdict(config),
                "settings": [asdict(s) for s in fitconfig.SETTINGS],
            }
        )

    @app.route("/api/image")
    def get_image():
        if g.result is None:
            return jsonify({"error": "No frame available"}), 400
        return jpeg_response(g.result.frame)

    @app.route("/api/image_within_rect")
    def get_image_within_rect():
        if g.result is None:
            return jsonify({"error": "No frame available"}), 400
        if g.result.options.rect is None:
            return jsonify({"error": "No rectangle defined"}), 400
        cropped = crop_rect(g.result.frame, g.result.roi, g.result.options.rect)
        if cropped is None:
            return jsonify({"error": "Rectangle outside of ROI"}), 400
        return jpeg_response(cropped[0])

    @app.route("/api/set_rect", methods=["POST"])
    def set_rect():
        coords, error = parse_coords()
        if error:
            return error
        session.set_region('rect_sensor', coords)
        return jsonify({"message": "Rectangle set successfully", "coords": coords})

    @app.route("/api/set_fit_rect", methods=["POST"])
    def set_fit_rect():
        coords, error = parse_coords()
        if error:
            return error
        session.set_region('fit_rect_sensor', coords)
        return jsonify({"message": "Fitting rectangle set successfully", "coords": coords})

    @app.route("/api/clear_rect", methods=["POST"])
    def clear_rect():
        session.set_region('rect_sensor', None)
        return jsonify({"message": "Rectangle cleared"})

    @app.route("/api/clear_fit_rect", methods=["POST"])
    def clear_fit_rect():
        session.set_region('fit_rect_sensor', None)
        return jsonify({"message": "Fitting rectangle cleared"})

    @app.route("/")
    def index():
        if g.snapshot.angular is not None:
            return render_template('angular.html')
        rows = "\n".join(
            f'<div class="endpoint"><span class="method">{m}</span> '
            f"<code>{path}</code><p>{desc}</p></div>"
            for m, path, desc in ENDPOINTS
        )
        return f"""<!DOCTYPE html>
<html><head><title>Beam Profiler Server</title><style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  .endpoint {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px; }}
  .method {{ color: #007acc; font-weight: bold; }}
  pre {{ background: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }}
</style></head><body>
<h1>Beam Profiler Web Server</h1>
<h2>Available Endpoints:</h2>
{rows}
<h2>Examples:</h2>
<pre>
curl http://localhost:5000/api/status
curl -X POST -H "Content-Type: application/json" \\
     -d '{{"coords": [100, 100, 300, 300]}}' http://localhost:5000/api/set_rect
curl http://localhost:5000/api/rect_stats
curl http://localhost:5000/api/image -o current_image.jpg
</pre>
</body></html>"""

    return app
