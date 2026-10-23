"""Quaternion and rotation helpers.

Quaternions are (w, x, y, z), the same order MuJoCo uses. A quaternion `q`
describes the robot's orientation: it rotates vectors from the body frame
into the world frame.

This is plain NumPy on purpose. The autonomy code must not depend on the
simulator, so it can't borrow MuJoCo's math functions.
"""

import numpy as np


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """Rotation matrix -> quaternion (Shepperd's method, numerically stable)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_to_rotvec(q):
    """Quaternion -> rotation vector (axis * angle), always the short way round.

    q and -q are the same orientation; flipping to w >= 0 makes sure we
    never ask the robot to rotate 300 degrees when 60 would do.
    """
    q = np.asarray(q, float) / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return 2 * v
    return v / s * (2 * np.arctan2(s, q[0]))


def quat_angle(a, b):
    """Angle in radians of the rotation that takes orientation a to orientation b."""
    return float(np.linalg.norm(quat_to_rotvec(quat_mul(quat_conj(a), b))))


def look_at_quat(direction, up_hint=(0.0, 0.0, 1.0)):
    """Orientation whose body +x axis (where the camera points) faces `direction`.

    `up_hint` picks the roll: body +z ends up as close to it as possible.
    Passing the robot's current body z keeps roll unchanged, which avoids
    pointless spinning about the line of sight.
    """
    x = np.asarray(direction, float)
    x = x / np.linalg.norm(x)
    y = np.cross(up_hint, x)
    if np.linalg.norm(y) < 1e-6:  # looking straight along the hint: any roll will do
        y = np.cross([1.0, 0, 0] if abs(x[0]) < 0.9 else [0, 1.0, 0], x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return mat_to_quat(np.column_stack([x, y, z]))
