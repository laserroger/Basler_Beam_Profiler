"""Spot geometry and display smoothing, independent of any window or camera."""
import numpy as np
from .spots import SpotArray
from .grid import classify_grid
from .shear import fit_grid_shear, unavailable


class SpotStatistics:
    def __init__(self, ema=.8):
        self.ema = ema
        self.std_dx_ema = self.std_dy_ema = self.sigma_std_ema = 0.
        self._stats_grid_mode = None
        self.reset()

    def reset(self):
        self.stats_dx = self.stats_dy = self.stats_sigma = np.array([0.0])
        self.rows, self.columns, self.grid_stats = [], [], {}
        self.grid_shear = unavailable("Need at least two classified rows and columns")

    def update(self, spots: SpotArray, saturated=None):
        grid_mode = False
        if len(spots) > 1:
            self.rows, self.columns, self.grid_stats = classify_grid(spots, eps=20)
            self.grid_shear = (unavailable("Saturated spots: reduce exposure")
                               if saturated is not None and np.any(saturated)
                               else fit_grid_shear(self.rows, self.columns))
            # H controls overlays, not the geometry used for measurements.
            grid_mode = len(self.rows) >= 2 and len(self.columns) >= 2
            if grid_mode:
                # Never connect the end of one row to a spot in another row.
                dx = [np.diff(row.x) for row in self.rows if len(row) > 1]
                dy = [np.diff(col.y) for col in self.columns if len(col) > 1]
                self.stats_dx = np.concatenate(dx) if dx else np.array([0.0])
                self.stats_dy = np.concatenate(dy) if dy else np.array([0.0])
            else:
                order = np.argsort(spots.x)
                self.stats_dx = np.diff(spots.x[order])
                self.stats_dy = np.diff(spots.y[order])
            self.stats_sigma = spots.sigma
        else:
            self.reset()
        # Do not carry the previous mode's incompatible statistics into this one.
        smoothing = self.ema if getattr(self, '_stats_grid_mode', None) == grid_mode else 0.0
        self._stats_grid_mode = grid_mode
        self.std_dx_ema = smoothing * self.std_dx_ema + (1 - smoothing) * float(np.std(self.stats_dx))
        self.std_dy_ema = smoothing * self.std_dy_ema + (1 - smoothing) * float(np.std(self.stats_dy))
        self.sigma_std_ema = (
            self.ema * self.sigma_std_ema + (1 - self.ema) * float(np.std(self.stats_sigma))
        )

    @staticmethod
    def curvature(deltas) -> float:
        """Mean second difference of the spot spacings (sign = bow direction)."""
        return float(np.mean(np.diff(deltas))) if len(deltas) > 1 else 0.0

    def summary(self, pixel_size):
        um = pixel_size * 1e6
        stats = {
            "grid_shear": self.grid_shear,
            "basic_stats": {
                "dx_std": float(np.std(self.stats_dx)),
                "dy_std": float(np.std(self.stats_dy)),
                "dx_mean_um": float(np.mean(self.stats_dx)) * um,
                "dy_mean_um": float(np.mean(self.stats_dy)) * um,
                "sigma_std": self.sigma_std_ema,
            }
        }
        if self.grid_stats:
            grid = {
                axis: {
                    k: v for k, v in axis_stats.items() if isinstance(v, (int, float))
                }
                for axis, axis_stats in self.grid_stats.items()
            }
            for axis in grid:
                for key in list(grid[axis]):
                    if key.startswith("avg_"):
                        grid[axis][f"{key}_um"] = grid[axis][key] * um
            stats["grid_stats"] = grid
            stats["row_col_counts"] = {
                "num_rows": len(self.rows),
                "num_columns": len(self.columns),
            }
        return stats
