import numpy as np

from space_autonomy.controllers.pose_controller import PoseController
from space_autonomy.core.rotations import quat_angle, quat_from_axis_angle
from space_autonomy.core.types import VehicleSpec, Wrench
from space_autonomy.sim.world import SpaceSim


def test_vehicle_inertia_matches_mujoco():
    v = VehicleSpec()
    sim = SpaceSim(v)
    assert np.allclose(sim.model.body("inspector").inertia, v.inertia)


def test_robot_drifts_without_thrust():
    """Newton's first law: no gravity, no drag, so velocity stays constant."""
    sim = SpaceSim(VehicleSpec())
    sim.data.qvel[:3] = [0.1, 0, 0]
    sim.step(Wrench.zero(), 10.0)
    assert np.allclose(sim.state().position, [1.0, 0, 0], atol=1e-6)


def test_reaches_pose_target():
    v = VehicleSpec()
    sim = SpaceSim(v, start_pos=(0, 0, 0), start_quat=(1, 0, 0, 0))
    ctrl = PoseController(v)
    goal = np.array([3.0, -2.0, 1.0])
    goal_q = quat_from_axis_angle([1, 1, 0], np.radians(120))
    while sim.t < 60:
        sim.step(ctrl.command(sim.state(), goal, goal_q), 0.05)
    s = sim.state()
    assert np.linalg.norm(s.position - goal) < 0.05
    assert np.degrees(quat_angle(s.quat, goal_q)) < 1.0
    assert np.linalg.norm(s.velocity) < 0.01


def test_thrust_respects_limits():
    v = VehicleSpec()
    sim = SpaceSim(v)
    ctrl = PoseController(v)
    cmd = ctrl.command(sim.state(), [100.0, 0, 0], quat_from_axis_angle([0, 0, 1], 3.0))
    assert np.abs(cmd.force).max() <= v.max_force + 1e-12
    assert np.abs(cmd.torque).max() <= v.max_torque + 1e-12
