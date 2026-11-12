"""D4sigma diameters and independent traces for multiple selected spots."""
from collections import deque
import sys
import numpy as np


def spot_sizes(result):
    spots = result.spots
    order = np.lexsort((spots.y, spots.x))
    major = 2 * spots.sigma_0[order] * result.pixel_size * 1e6
    minor = 2 * spots.sigma_1[order] * result.pixel_size * 1e6
    saturated = result.saturated[order]
    valid = np.isfinite(major) & np.isfinite(minor) & (major > 0) & (minor > 0) & ~saturated
    def values(a):
        return [float(v) if np.isfinite(v) else None for v in a]
    return dict(
        timestamp=result.timestamp, convention='D4sigma', units='um', roi=list(result.roi),
        spot_count=len(spots), valid_count=int(valid.sum()),
        index=list(range(1, len(spots) + 1)),
        x_sensor_px=values(spots.x[order] + result.roi[2]),
        y_sensor_px=values(spots.y[order] + result.roi[3]),
        major_um=values(major), minor_um=values(minor),
        valid=valid.tolist(), saturated=saturated.tolist(),
    )


def trace_color(trace_id):
    """Stable colors survive additions/removals of other selected spots."""
    palette = ('#41bfff', '#ffc35a', '#dc8fff', '#6fe3ad', '#ff7e91', '#a6bfff')
    return palette[(trace_id-1) % len(palette)]


# Keep current records + a published previous snapshot below 10 MB combined.
# Each side has a 4 MB budget, leaving room for containers and handoff overhead.
HISTORY_STORAGE_BYTES = 4_000_000
MAX_HISTORY_SAMPLES = 12_000  # shared across every selected spot


def _record_bytes(sample):
    return sys.getsizeof(sample) + sum(sys.getsizeof(v) for v in sample.values())


class SizeMonitor:
    """Independent, bounded histories for explicitly selected sensor anchors.

    Clicks toggle individual spots. Association gates shrink for close beams;
    missing, ambiguous and saturated detections leave gaps in that trace alone.
    """
    def __init__(self):
        self.samples = {}
        self.targets = {}
        self.context = None
        self.latest = None
        self.gate_px = 40.
        self._next_id = 1
        self.time_origin = None
        self.window_seconds = 60.
        self._sample_bytes = 0

    def history_memory_bytes(self):
        # Include the reference lists produced for the published snapshot.
        return (self._sample_bytes + sys.getsizeof(self.samples)
                + sum(sys.getsizeof(samples) + 56 + 8*len(samples)
                      for samples in self.samples.values()))

    def _remove_target(self, key):
        self._sample_bytes -= sum(_record_bytes(h) for h in self.samples[key])
        del self.targets[key], self.samples[key]

    def _drop_oldest(self, samples):
        self._sample_bytes -= _record_bytes(samples.popleft())

    def _limit_history(self):
        count = sum(len(samples) for samples in self.samples.values())
        while count > MAX_HISTORY_SAMPLES or self.history_memory_bytes() > HISTORY_STORAGE_BYTES:
            candidates = [(samples[0]['timestamp'],key) for key,samples in self.samples.items() if samples]
            if not candidates:
                break
            key = min(candidates)[1]
            self._drop_oldest(self.samples[key])
            count -= 1

    def reset_time(self):
        for samples in self.samples.values():
            samples.clear()
        self.time_origin = None
        self._sample_bytes = 0

    def set_window(self, seconds):
        seconds = float(seconds)
        if not np.isfinite(seconds) or not 1 <= seconds <= 600:
            raise ValueError('Time window must be between 1 and 600 seconds')
        self.window_seconds = seconds

    def select(self, position):
        if position is None:
            self.time_origin = None
            self.targets.clear()
            self.samples.clear()
            self._sample_bytes = 0
            return True
        x, y = position
        data = self.latest
        if data is None:
            return False
        distance = np.hypot(np.asarray(data['x_sensor_px'], dtype=float)-x,
                            np.asarray(data['y_sensor_px'], dtype=float)-y)
        distance = np.where(np.isfinite(distance), distance, np.inf)
        i = int(np.argmin(distance)) if len(distance) else None
        if i is None or distance[i] > self.gate_px:
            # A missing selected spot can still be deselected by its ring.
            nearby = [(np.hypot(t['x_sensor_px']-x,t['y_sensor_px']-y), k)
                      for k,t in self.targets.items()]
            nearby = [(d,k) for d,k in nearby if d <= self.targets[k]['gate_px']]
            if nearby:
                key = min(nearby)[1]
                self._remove_target(key)
                return True
            return False
        anchor = (data['x_sensor_px'][i], data['y_sensor_px'][i])
        for key, target in list(self.targets.items()):
            if np.hypot(target['x_sensor_px']-anchor[0],target['y_sensor_px']-anchor[1]) <= target['gate_px']:
                self._remove_target(key)
                return True
        neighbours = np.hypot(np.asarray(data['x_sensor_px'], dtype=float)-anchor[0],
                              np.asarray(data['y_sensor_px'], dtype=float)-anchor[1])
        neighbours[i] = np.inf
        gate = min(self.gate_px, float(np.nanmin(neighbours))/2.1)
        key = self._next_id
        self._next_id += 1
        self.targets[key] = dict(id=key,x_sensor_px=anchor[0],y_sensor_px=anchor[1],
                                 gate_px=gate,color=trace_color(key))
        self.samples[key] = deque()
        return True

    def update(self, result, context=None, z_um=None):
        if context != self.context:
            self.reset_time()
            self.context = context
        data = spot_sizes(result)
        if self.targets and self.time_origin is None:
            self.time_origin = result.timestamp
        data["time_origin"] = self.time_origin
        data["window_seconds"] = self.window_seconds
        self.latest = data
        matches = {}
        claims = {}
        for key, target in self.targets.items():
            distances = np.hypot(np.asarray(data['x_sensor_px'], dtype=float)-target['x_sensor_px'],
                                 np.asarray(data['y_sensor_px'], dtype=float)-target['y_sensor_px'])
            candidates = np.flatnonzero(distances <= target['gate_px'])
            matches[key] = candidates
            if len(candidates) == 1:
                i = int(candidates[0])
                claims[i] = claims.get(i, 0) + 1
        traces = []
        for key, target in self.targets.items():
            candidates = matches[key]
            i = int(candidates[0]) if len(candidates) == 1 else None
            ambiguous = len(candidates) > 1 or (i is not None and claims[i] > 1)
            valid = i is not None and not ambiguous and data['valid'][i]
            major = data['major_um'][i] if valid else None
            minor = data['minor_um'][i] if valid else None
            reason = ('valid' if valid else 'ambiguous detections' if ambiguous
                      else 'saturated or invalid fit' if i is not None else 'spot not detected')
            samples = self.samples[key]
            sample=dict(timestamp=data['timestamp'],major_um=major,minor_um=minor,z_um=z_um)
            samples.append(sample)
            self._sample_bytes += _record_bytes(sample)
            while samples and data['timestamp'] - samples[0]['timestamp'] > self.window_seconds:
                self._drop_oldest(samples)
            traces.append(dict(target,detected_index=i,valid=bool(valid),reason=reason,
                               major_um=major,minor_um=minor))
        self._limit_history()
        for trace in traces:
            trace['history'] = list(self.samples[trace['id']])
        data['history_storage_bytes'] = self.history_memory_bytes()
        data['history_memory_limit_bytes'] = 10_000_000
        data['traces'] = traces
        return data
