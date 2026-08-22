"""Columnar container for detected spots.

The fit produces one numpy array per quantity instead of one dict per spot:
on an 80x80 array (6400 spots) building the dicts costs more time than the
Gaussian fit itself.  `SpotArray` keeps the columns and materialises a dict
only when something indexes or iterates it, so the hot paths (grid
classification, statistics, overlay drawing) stay vectorised while older
`spot["x"]` code keeps working unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FIELDS = ("x", "y", "sigma_0", "sigma_1", "vec_0", "vec_1", "I0", "I0_weighted")


class SpotArray(Sequence):
    """Sequence of spots stored as columns.

    Scalar columns are (n,) float64; `vec_0`/`vec_1` are (n, 2).  Indexing with
    an int yields the legacy dict (including the derived "sigma" key); slices
    and index arrays yield another SpotArray.
    """

    __slots__ = ("x", "y", "sigma_0", "sigma_1", "vec_0", "vec_1", "I0", "I0_weighted", "_sigma")

    def __init__(self, x, y, sigma_0, sigma_1, vec_0, vec_1, I0, I0_weighted):
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.sigma_0 = np.ascontiguousarray(sigma_0, dtype=np.float64)
        self.sigma_1 = np.ascontiguousarray(sigma_1, dtype=np.float64)
        self.vec_0 = np.ascontiguousarray(vec_0, dtype=np.float64).reshape(-1, 2)
        self.vec_1 = np.ascontiguousarray(vec_1, dtype=np.float64).reshape(-1, 2)
        self.I0 = np.ascontiguousarray(I0, dtype=np.float64)
        self.I0_weighted = np.ascontiguousarray(I0_weighted, dtype=np.float64)
        self._sigma = None

    # ------------------------------ construction -------------------------- #
    @classmethod
    def empty(cls) -> SpotArray:
        z, v = np.empty(0), np.empty((0, 2))
        return cls(z, z, z, z, v, v, z, z)

    @classmethod
    def from_dicts(cls, spots) -> SpotArray:
        spots = list(spots)
        if not spots:
            return cls.empty()
        col = lambda k: np.array([s[k] for s in spots], dtype=np.float64)  # noqa: E731
        return cls(
            col("x"), col("y"), col("sigma_0"), col("sigma_1"),
            np.array([s["vec_0"] for s in spots], dtype=np.float64),
            np.array([s["vec_1"] for s in spots], dtype=np.float64),
            col("I0"), col("I0_weighted"),
        )

    @classmethod
    def concat(cls, parts) -> SpotArray:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        if len(parts) == 1:
            return parts[0]
        cat = lambda k: np.concatenate([getattr(p, k) for p in parts])  # noqa: E731
        return cls(*(cat(f) for f in FIELDS))

    # ------------------------------ sequence ------------------------------ #
    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            if i < 0:
                i += len(self)
            if not 0 <= i < len(self):
                raise IndexError(i)
            return {
                "x": self.x[i], "y": self.y[i],
                "sigma_0": self.sigma_0[i], "sigma_1": self.sigma_1[i],
                "sigma": self.sigma[i],
                "vec_0": self.vec_0[i], "vec_1": self.vec_1[i],
                "I0": self.I0[i], "I0_weighted": self.I0_weighted[i],
            }
        return SpotArray(*(getattr(self, f)[i] for f in FIELDS))

    # ------------------------------ derived ------------------------------- #
    @property
    def sigma(self) -> np.ndarray:
        """Geometric mean width, sqrt(sigma_0 * sigma_1)."""
        if self._sigma is None:
            self._sigma = np.sqrt(self.sigma_0 * self.sigma_1)
        return self._sigma

    @property
    def angle(self) -> np.ndarray:
        """Orientation of the major axis, degrees."""
        return np.degrees(np.arctan2(self.vec_0[:, 1], self.vec_0[:, 0]))

    def sorted_by(self, key: str) -> SpotArray:
        return self[np.argsort(getattr(self, key), kind="stable")]

    def to_json(self) -> list[dict]:
        """Plain-Python mirror for the HTTP API (built on demand, not per frame)."""
        return [
            {
                "x": x, "y": y, "sigma_0": s0, "sigma_1": s1, "sigma": s,
                "vec_0": [v0x, v0y], "vec_1": [v1x, v1y],
                "I0": i0, "I0_weighted": i0w,
            }
            for x, y, s0, s1, s, (v0x, v0y), (v1x, v1y), i0, i0w in zip(
                self.x.tolist(), self.y.tolist(), self.sigma_0.tolist(),
                self.sigma_1.tolist(), self.sigma.tolist(),
                self.vec_0.tolist(), self.vec_1.tolist(),
                self.I0.tolist(), self.I0_weighted.tolist(),
            )
        ]
