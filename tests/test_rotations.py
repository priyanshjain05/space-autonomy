import numpy as np

from space_autonomy.core.rotations import (look_at_quat, mat_to_quat, quat_angle, quat_from_axis_angle,
                                           quat_mul, quat_to_mat, quat_to_rotvec)

rng = np.random.default_rng(0)


def random_quat():
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_matrix_roundtrip():
    for _ in range(100):
        q = random_quat()
        q2 = mat_to_quat(quat_to_mat(q))
        assert quat_angle(q, q2) < 1e-9


def test_composition_matches_matrices():
    a, b = random_quat(), random_quat()
    assert np.allclose(quat_to_mat(quat_mul(a, b)), quat_to_mat(a) @ quat_to_mat(b))


def test_rotvec_takes_short_way():
    q = quat_from_axis_angle([0, 0, 1], np.radians(300))
    assert np.allclose(quat_to_rotvec(q), [0, 0, np.radians(-60)])


def test_look_at_points_body_x_along_direction():
    for d in ([1, 2, 3], [0, 0, -1], [0, 0, 1], [-5, 0.1, 0]):
        R = quat_to_mat(look_at_quat(d))
        assert np.allclose(R[:, 0], np.array(d) / np.linalg.norm(d))
        assert np.isclose(np.linalg.det(R), 1)
