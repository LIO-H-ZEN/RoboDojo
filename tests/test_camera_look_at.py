import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from utils.rotations import look_at_to_usd_camera_quat


def _rotation_matrix(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz)
    return Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()


def test_usd_camera_look_at_maps_negative_z_to_target_and_positive_y_to_up():
    eye = np.array([0.0, -0.20, 1.215])
    target = np.array([0.0, -0.07, 0.845])
    quat = look_at_to_usd_camera_quat(eye, target, [0.0, 0.0, 1.0])
    rotation = _rotation_matrix(quat)

    expected_forward = target - eye
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(rotation @ [0.0, 0.0, -1.0], expected_forward, atol=1e-6)
    assert float((rotation @ [0.0, 1.0, 0.0]) @ np.array([0.0, 0.0, 1.0])) > 0.0
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    ("eye", "target", "up", "message"),
    [
        ([0, 0, 0], [0, 0, 0], [0, 0, 1], "eye and target"),
        ([0, 0, 0], [1, 0, 0], [0, 0, 0], "up vector must be non-zero"),
        ([0, 0, 0], [1, 0, 0], [1, 0, 0], "must not be parallel"),
    ],
)
def test_usd_camera_look_at_rejects_degenerate_geometry(eye, target, up, message):
    with pytest.raises(ValueError, match=message):
        look_at_to_usd_camera_quat(eye, target, up)
