"""Pose canonicalization helpers for weakly observable object axes.

For a near-cylindrical object, RGB-D registration can estimate the centre and
the long-axis direction correctly while the arbitrary rotation about that
axis drifts.  ``lock_axial_rotation`` removes only that unobservable degree
of freedom relative to the previous published pose; translation and tilt are
left unchanged.
"""

from __future__ import annotations

import math

import numpy as np


def _unit_axis(axis: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    value = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm < 1e-8:
        raise ValueError("axial symmetry axis must be a finite non-zero 3-vector")
    return value / norm


def _axis_basis(axis: np.ndarray) -> np.ndarray:
    """Return a right-handed basis whose third column is ``axis``."""
    candidates = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    reference = min(candidates, key=lambda item: abs(float(np.dot(item, axis))))
    first = reference - float(np.dot(reference, axis)) * axis
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, axis))


def _local_axis_rotation(theta: float) -> np.ndarray:
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )


def lock_axial_rotation(
    current: np.ndarray,
    reference: np.ndarray,
    axis: np.ndarray | list[float] | tuple[float, ...] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Canonicalize ``current`` under rotation about an object-frame axis.

    The returned transform is ``current`` with a right-multiplied rotation
    around ``axis`` chosen to minimize its rotation error to ``reference``.
    This is the correct equivalence operation for a centered object mesh.
    """
    current = np.asarray(current, dtype=float).reshape(4, 4)
    reference = np.asarray(reference, dtype=float).reshape(4, 4)
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("pose contains a non-finite value")
    unit = _unit_axis(axis)
    basis = _axis_basis(unit)
    relative = reference[:3, :3].T @ current[:3, :3]
    relative_local = basis.T @ relative @ basis

    # Maximize trace(relative * R_axis(theta)).  The signs below cancel a
    # continuously drifting axial phase (for example, a cylindrical bottle's
    # arbitrary yaw) while preserving the axis direction and translation.
    cosine_term = float(relative_local[0, 0] + relative_local[1, 1])
    sine_term = float(relative_local[0, 1] - relative_local[1, 0])
    theta = math.atan2(sine_term, cosine_term)
    correction = basis @ _local_axis_rotation(theta) @ basis.T

    result = current.copy()
    result[:3, :3] = current[:3, :3] @ correction
    return result

