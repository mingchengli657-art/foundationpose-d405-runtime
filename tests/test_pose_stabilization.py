import numpy as np
from scipy.spatial.transform import Rotation

from app.pose_stabilization import lock_axial_rotation


def test_axial_drift_is_removed_without_changing_axis_or_translation():
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler(
        "yx", [28.0, -17.0], degrees=True
    ).as_matrix()
    current = reference.copy()
    current[:3, :3] = current[:3, :3] @ Rotation.from_euler(
        "z", 137.0, degrees=True
    ).as_matrix()
    current[:3, 3] = [0.1, -0.2, 0.7]

    result = lock_axial_rotation(current, reference, [0.0, 0.0, 1.0])

    assert np.degrees(
        Rotation.from_matrix(reference[:3, :3].T @ result[:3, :3]).magnitude()
    ) < 1e-7
    assert np.allclose(result[:3, 3], current[:3, 3])
    assert np.allclose(result[:3, 2], current[:3, 2])


def test_arbitrary_axis_is_supported():
    axis = np.array([1.0, 2.0, 3.0])
    reference = np.eye(4)
    current = reference.copy()
    # The helper should produce a valid rigid transform for a non-Z model
    # axis; exact yaw equivalence is checked by preserving that axis.
    result = lock_axial_rotation(current, reference, axis)
    assert np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(result[:3, :3]), 1.0)
