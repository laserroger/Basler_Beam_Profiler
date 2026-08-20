import numpy as np

# ------------------------------ ROI MODEL ------------------------------- #


class ROIModel:
    """Maintain ROI as (scale, aspect, centre) → tuple understood by Basler."""

    def __init__(self, sensor_w: int, sensor_h: int):
        self.full_w = sensor_w
        self.full_h = sensor_h
        self.scale = 1.0  # fraction of *width*
        self.aspect = sensor_w / sensor_h  # w / h  (initial native)
        self.cx = sensor_w / 2  # centre in pixel coords
        self.cy = sensor_h / 2

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def size(self):
        w = max(1, int(self.full_w * self.scale))
        h = max(1, int(w / self.aspect))
        # shrink if we spill vertically
        if h > self.full_h:
            h = self.full_h
            w = int(h * self.aspect)
        return w, h

    @property
    def tuple(self):
        """Return (w, h, offset_x, offset_y) as expected by Basler."""
        w, h = self.size
        ox = int(self.cx - w / 2)
        oy = int(self.cy - h / 2)
        # clamp
        ox = max(0, min(ox, self.full_w - w))
        oy = max(0, min(oy, self.full_h - h))
        # recalc centre to honour clamping
        self.cx = ox + w / 2
        self.cy = oy + h / 2
        return w, h, ox, oy

    def display_to_sensor(self, disp_x, disp_y, display_size, pad_l, pad_t):
        w_roi, h_roi, ox, oy = self.tuple
        # remove padding
        x_no_pad = disp_x - pad_l
        y_no_pad = disp_y - pad_t
        # get real image size
        scale = display_size / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        # convert to relative coordinates
        r_x = x_no_pad / view_w if view_w > 0 else 0
        r_y = y_no_pad / view_h if view_h > 0 else 0
        # convert to sensor coordinates
        sensor_x = ox + r_x * w_roi
        sensor_y = oy + r_y * h_roi
        return sensor_x, sensor_y

    def sensor_to_display(self, sensor_x, sensor_y, display_size, pad_l, pad_t):
        w_roi, h_roi, ox, oy = self.tuple
        #  convert to relative coordinates
        r_x = (sensor_x - ox) / w_roi if w_roi > 0 else 0
        r_y = (sensor_y - oy) / h_roi if h_roi > 0 else 0
        # get real image size
        scale = display_size / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        # convert to display coordinates
        disp_x = int(r_x * view_w) + pad_l
        disp_y = int(r_y * view_h) + pad_t
        return disp_x, disp_y

    def coord_in_roi_to_sensor(self, x, y):
        """Convert coordinates in the ROI to sensor coordinates."""
        w_roi, h_roi, ox, oy = self.tuple
        # convert to relative coordinates
        r_x = x / w_roi if w_roi > 0 else 0
        r_y = y / h_roi if h_roi > 0 else 0
        # convert to sensor coordinates
        sensor_x = ox + r_x * w_roi
        sensor_y = oy + r_y * h_roi
        return sensor_x, sensor_y

    # -------------------------- interactions --------------------------- #
    def keep_point_fixed(
        self, sensor_x: float, sensor_y: float, new_scale=None, new_aspect=None
    ):
        """Update scale/aspect so that *sensor_x, sensor_y* remain
        at the *same absolute position* in the sensor afterwards.
        Additionally, when changing *aspect*, keep the longest edge length
        unchanged so zoom level feels intuitive (requirement #4)."""
        # Current ROI geometry
        w_old, h_old, ox_old, oy_old = self.tuple
        long_old = max(w_old, h_old)
        r_x = (sensor_x - ox_old) / w_old if w_old else 0.5
        r_y = (sensor_y - oy_old) / h_old if h_old else 0.5

        # First apply new aspect so we can compensate scale later
        if new_aspect is not None:
            new_aspect = max(0.1, min(new_aspect, 20.0))
            # Predict size with *current* scale
            w_tmp = self.full_w * self.scale
            h_tmp = w_tmp / new_aspect
            long_tmp = max(w_tmp, h_tmp)
            if long_tmp > 0:
                self.scale *= long_old / long_tmp  # keep longest edge constant
            self.aspect = new_aspect

        # Apply zoom afterwards in case both happen in same event
        if new_scale is not None:
            self.scale = max(0.02, min(new_scale, 1.0))

        # Re‑compute ROI and recalc centre while clamping
        w_new, h_new = self.size
        ox_new = sensor_x - r_x * w_new
        oy_new = sensor_y - r_y * h_new
        self.cx = np.clip(ox_new + w_new / 2, w_new / 2, self.full_w - w_new / 2)
        self.cy = np.clip(oy_new + h_new / 2, h_new / 2, self.full_h - h_new / 2)


# --------------------------- STATISTICS ------------------------------- #


def classify_dots_grid(spots, eps=None, min_samples=2):
    """
    使用基于邻近距离分析的聚类算法对点阵进行行列分类

    参数:
        spots: 包含点信息的列表，每个点需要有'x'和'y'键
        eps: 相邻聚类的最大距离，None表示自动确定
        min_samples: 形成一个有效行/列所需的最小点数

    返回:
        rows: 按行分类的点列表
        columns: 按列分类的点列表
        stats: 包含行列统计信息的字典
    """
    if not spots:
        return [], [], {}

    # 提取坐标
    xs = np.array([s["x"] for s in spots])
    ys = np.array([s["y"] for s in spots])

    # ------------------------- 基于距离分析的行列聚类 -------------------------

    # 分析X方向（查找列）
    sorted_x_indices = np.argsort(xs)
    sorted_xs = xs[sorted_x_indices]

    # 计算相邻X坐标差值
    x_gaps = np.diff(sorted_xs)

    # 自动确定聚类阈值
    if eps is None:
        # 使用基于分布的自适应阈值
        if len(x_gaps) > 0:
            median_x_gap = np.median(x_gaps)
            # Tukey方法识别异常大间隔
            q75, q25 = np.percentile(x_gaps, [85, 15])
            iqr = q75 - q25
            x_threshold = q75 + (70 / 15) * iqr

            # 使用最小阈值确保鲁棒性
            x_threshold = max(x_threshold, 2.0 * median_x_gap)
        else:
            x_threshold = 1.0  # 默认值
    else:
        x_threshold = eps

    # 基于间隔识别列边界
    x_break_points = np.where(x_gaps > x_threshold)[0]

    # 根据边界分配列
    columns = []
    start_idx = 0

    for break_point in x_break_points:
        end_idx = break_point + 1
        if end_idx - start_idx >= min_samples:
            col_indices = sorted_x_indices[start_idx:end_idx]
            col_points = [spots[i] for i in col_indices]
            columns.append(col_points)
        start_idx = end_idx

    # 处理最后一列
    if len(sorted_xs) - start_idx >= min_samples:
        col_indices = sorted_x_indices[start_idx:]
        col_points = [spots[i] for i in col_indices]
        columns.append(col_points)

    # 分析Y方向（查找行）
    sorted_y_indices = np.argsort(ys)
    sorted_ys = ys[sorted_y_indices]

    # 计算相邻Y坐标差值
    y_gaps = np.diff(sorted_ys)

    # 自动确定聚类阈值
    if eps is None:
        # 使用基于分布的自适应阈值
        if len(y_gaps) > 0:
            median_y_gap = np.median(y_gaps)
            # Tukey方法识别异常大间隔
            q75, q25 = np.percentile(y_gaps, [85, 15])
            iqr = q75 - q25
            y_threshold = q75 + (70 / 15) * iqr

            # 使用最小阈值确保鲁棒性
            y_threshold = max(y_threshold, 2.0 * median_y_gap)
        else:
            y_threshold = 1.0  # 默认值
    else:
        y_threshold = eps

    # 基于间隔识别行边界
    y_break_points = np.where(y_gaps > y_threshold)[0]

    # 根据边界分配行
    rows = []
    start_idx = 0

    for break_point in y_break_points:
        end_idx = break_point + 1
        if end_idx - start_idx >= min_samples:
            row_indices = sorted_y_indices[start_idx:end_idx]
            row_points = [spots[i] for i in row_indices]
            rows.append(row_points)
        start_idx = end_idx

    # 处理最后一行
    if len(sorted_ys) - start_idx >= min_samples:
        row_indices = sorted_y_indices[start_idx:]
        row_points = [spots[i] for i in row_indices]
        rows.append(row_points)

    # ----------------------- 对每行和每列内点进行排序 --------------------------
    # 每行内按x坐标排序
    for row in rows:
        row.sort(key=lambda s: s["x"])

    # 每列内按y坐标排序
    for col in columns:
        col.sort(key=lambda s: s["y"])

    # -------------------------- 计算统计信息 -----------------------------
    stats = {
        "rows": {"mean_dx": [], "std_x": [], "std_y": []},
        "columns": {"mean_dy": [], "std_y": [], "std_x": []},
    }

    # 计算每行的统计信息
    for row in rows:
        if len(row) >= 2:
            row_xs = np.array([s["x"] for s in row])
            row_xs.sort()
            row_dxs = np.diff(row_xs)

            stats["rows"]["mean_dx"].append(float(row_dxs.mean()))
            stats["rows"]["std_x"].append(float(row_dxs.std()))

            # calculate the deviation in y for each row
            row_ys = np.array([s["y"] for s in row])
            stats["rows"]["std_y"].append(float(row_ys.std()))

    # 计算每列的统计信息
    for col in columns:
        if len(col) >= 2:
            col_ys = np.array([s["y"] for s in col])
            col_ys.sort()
            col_dys = np.diff(col_ys)

            stats["columns"]["mean_dy"].append(float(col_dys.mean()))
            stats["columns"]["std_y"].append(float(col_dys.std()))

            # calculate the deviation in x for each column
            col_xs = np.array([s["x"] for s in col])
            stats["columns"]["std_x"].append(float(col_xs.std()))
    # ----------------------- 计算平均统计信息 -----------------------------

    # 计算所有行和列的平均统计信息
    for key in ["mean_dx", "std_x", "std_y"]:
        if stats["rows"][key]:
            stats["rows"][f"avg_{key}"] = float(np.mean(stats["rows"][key]))

    for key in ["mean_dy", "std_y", "std_x"]:
        if stats["columns"][key]:
            stats["columns"][f"avg_{key}"] = float(np.mean(stats["columns"][key]))

    return rows, columns, stats
