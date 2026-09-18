"""Moment-based 2D Gaussian estimation."""

from __future__ import annotations

import numpy as np


def moments_2d(X, Y, Z) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted mean and covariance of Z sampled on grids X, Y."""
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64).ravel()
    total = Z.sum()
    if total <= 0:
        raise ValueError("non-positive total intensity")
    mx = (X * Z).sum() / total
    my = (Y * Z).sum() / total
    dx, dy = X - mx, Y - my
    cov = (
        np.array(
            [
                [(Z * dx * dx).sum(), (Z * dx * dy).sum()],
                [(Z * dx * dy).sum(), (Z * dy * dy).sum()],
            ]
        )
        / total
    )
    return np.array([mx, my]), cov


def gaussian_exponent(X, Y, mu, cov):
    """Exponent of a 2D Gaussian (0 at the centre, -0.5 on the 1-sigma ellipse)."""
    inv = np.linalg.inv(cov)
    dx, dy = X - mu[0], Y - mu[1]
    return -0.5 * (inv[0, 0] * dx**2 + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy**2)


def gaussian_2d(X, Y, mu, cov):
    coeff = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return coeff * np.exp(gaussian_exponent(X, Y, mu, cov))
