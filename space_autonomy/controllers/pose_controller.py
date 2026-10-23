"""Pose controller: where the robot is vs. where it should be -> thruster wrench.

Both translation and rotation use the same two-stage ("cascaded") design:

    position error  --(x pos_gain)-->  desired velocity  (capped at max_speed)
    velocity error  --(x vel_gain)-->  desired acceleration  --(x mass)--> force

    attitude error  --(x att_gain)-->  desired spin rate (capped at max_rate)
    rate error      --(x rate_gain)--> desired angular accel --(x inertia)--> torque

Why two stages instead of one PD loop: the outer stage puts a hard cap on
speed, so the robot never builds up more momentum than it can brake with
its small thrusters. The inner stage is made ~5x faster than the outer one
so the outer stage can treat velocity as "whatever I ask for".

Gains are in 1/s: a gain of 0.15 means a position error decays with a time
constant of about 1/0.15 = 6.7 s once the robot is close to the target.
"""

from dataclasses import dataclass

import numpy as np

from ..core.rotations import quat_conj, quat_mul, quat_to_rotvec
from ..core.types import RobotState, VehicleSpec, Wrench


@dataclass(frozen=True)
class PoseGains:
    pos_gain: float = 0.15    # 1/s
    vel_gain: float = 0.8     # 1/s
    max_speed: float = 0.25   # m/s
    att_gain: float = 0.5     # 1/s
    rate_gain: float = 2.0    # 1/s
    max_rate: float = 0.3     # rad/s


def _cap_norm(v, limit):
    n = np.linalg.norm(v)
    return v if n <= limit else v * (limit / n)


def _fit_box(v, limit):
    """Scale a vector down so every component fits in [-limit, limit], keeping its direction.

    Clipping each axis separately would bend the thrust direction and push
    the robot off its intended line.
    """
    m = np.abs(v).max()
    return v if m <= limit else v * (limit / m)


class PoseController:
    def __init__(self, vehicle: VehicleSpec, gains: PoseGains = PoseGains()):
        self.vehicle = vehicle
        self.gains = gains

    def command(self, state: RobotState, target_pos, target_quat, cruise=False) -> Wrench:
        """Wrench (body frame) that drives the robot toward target_pos / target_quat.

        cruise=True is for pass-through waypoints: fly at full speed toward
        the point instead of slowing down to stop on it.
        """
        g, v = self.gains, self.vehicle
        R = state.rotation

        # Translation (world frame), then rotate into the body frame for the thrusters.
        err = np.asarray(target_pos, float) - state.position
        if cruise and np.linalg.norm(err) > 1e-9:
            v_des = g.max_speed * err / np.linalg.norm(err)
        else:
            v_des = _cap_norm(g.pos_gain * err, g.max_speed)
        accel = g.vel_gain * (v_des - state.velocity)
        force = _fit_box(R.T @ (v.mass * accel), v.max_force)

        # Rotation (body frame). q_err is the rotation from where we point now
        # to where we want to point, expressed in the body frame.
        q_err = quat_mul(quat_conj(state.quat), target_quat)
        w_des = _cap_norm(g.att_gain * quat_to_rotvec(q_err), g.max_rate)
        torque = _fit_box(v.inertia * g.rate_gain * (w_des - state.angular_velocity), v.max_torque)

        return Wrench(force, torque)
