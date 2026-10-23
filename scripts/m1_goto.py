"""M1: the free-flyer flies itself to a series of target poses in empty space.

Each target is a position plus an orientation, to exercise both the
translation and the rotation controllers.
    python scripts/m1_goto.py
"""

from pathlib import Path

import numpy as np

from space_autonomy.controllers.pose_controller import PoseController
from space_autonomy.core.rotations import quat_from_axis_angle, quat_mul
from space_autonomy.core.types import VehicleSpec
from space_autonomy.evaluation.harness import run_mission
from space_autonomy.evaluation.plots import plot_states, plot_trajectory
from space_autonomy.evaluation.report import format_report, save_run
from space_autonomy.planning.sequencer import Leg, LegSequencer
from space_autonomy.sim.world import SpaceSim

Z, Y = (0, 0, 1), (0, 1, 0)
yaw90 = quat_from_axis_angle(Z, np.pi / 2)
TARGETS = [
    ("forward 3 m", (3, 0, 0), np.array([1.0, 0, 0, 0])),
    ("left + yaw 90", (3, 3, 0), yaw90),
    ("up + pitch 45", (0, 3, 2), quat_mul(yaw90, quat_from_axis_angle(Y, -np.pi / 4))),
    ("home, flip 180", (0, 0, 0), quat_from_axis_angle(Z, np.pi)),
]

vehicle = VehicleSpec()
sim = SpaceSim(vehicle, start_pos=(0, 0, 0), start_quat=(1, 0, 0, 0))
legs = [Leg(name, [np.array(p, float)], goal_quat=q, dwell=2.0) for name, p, q in TARGETS]
autonomy = LegSequencer(legs, PoseController(vehicle))

result = run_mission(sim, autonomy, max_time=300, name="m1-goto")
out = save_run("results/m1", result)
plot_states(result.telemetry, out / "states.png", "M1: pose targets in free space")
plot_trajectory(result.telemetry, out / "trajectory.png",
                viewpoints=[(n, np.array(p, float)) for n, p, _ in TARGETS], title="M1: flown path")

for e in result.events:
    print(f"t={e['t']:7.1f}s  {e['event']:<17} {e['leg']}")
print()
print(format_report(result.summary))
print(f"\nOutputs in {out}/")
