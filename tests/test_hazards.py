import numpy as np

from space_autonomy.autonomy.hazards import Tracker, min_gap, timeline
from space_autonomy.controllers.pose_controller import PoseController
from space_autonomy.core.types import Detection, VehicleSpec
from space_autonomy.evaluation.batch import run_hazard
from space_autonomy.scenarios.alpha03 import with_anomalies
from space_autonomy.sim.world import SpaceSim


def test_prediction_matches_the_real_robot():
    """The predicted motion replays the controller, so it should track a real simulated flight closely,
    including the slow start from rest that a constant-speed guess gets wrong by 2x."""
    vehicle = VehicleSpec()
    sim = SpaceSim(vehicle, start_pos=(0, 0, 0), start_quat=(1, 0, 0, 0))
    ctrl = PoseController(vehicle)
    goal = np.array([3.0, 0, 0])
    at, arrive = timeline([0, 0, 0], [goal], 0.0)
    while sim.t < 30:
        sim.step(ctrl.command(sim.state(), goal, np.array([1.0, 0, 0, 0])), 0.05)
        assert np.linalg.norm(at(sim.t) - sim.state().position) < 0.03
    assert 20 < arrive < 30


def test_prediction_waits_and_pauses():
    at, _ = timeline([0, 0, 0], [[2, 0, 0], [2, 2, 0]], t0=0.0, wait=5.0, pauses={0: 10.0})
    assert np.allclose(at(4.0), [0, 0, 0])                     # still waiting
    stopped = [t for t in np.arange(5, 60, 0.5) if np.linalg.norm(at(t) - [2, 0, 0]) < 0.06]
    assert stopped[-1] - stopped[0] >= 10.0                    # holds at the first waypoint ...
    assert np.linalg.norm(at(59.0) - [2, 2, 0]) < 0.1          # ... then goes on to the next


def test_min_gap_head_on():
    at, _ = timeline([0, 0, 0], [], 0.0)  # standing still
    gap, when = min_gap(at, np.array([5.0, 0, 0]), np.array([-0.5, 0, 0]), gap_offset=0.5, t0=0.0, horizon=20.0)
    # Straight through the middle: the worst moment is when the centres coincide (t = 10 s),
    # where the "gap" is minus both radii.
    assert np.isclose(gap, -0.5, atol=0.01) and abs(when - 10.0) < 0.3


def test_tracker_recovers_velocity_from_noisy_sightings():
    rng = np.random.default_rng(0)
    tr = Tracker()
    v = np.array([0.1, -0.05, 0.02])
    for k in range(20):
        t = 0.2 * k
        tr.update([Detection(t, 7, np.array([1.0, 2, 3]) + v * t + rng.normal(0, 0.02, 3), 0.25)])
    p, v_est = tr.tracks[7].estimate()
    assert np.linalg.norm(v_est - v) < 0.03
    assert np.linalg.norm(p - (np.array([1.0, 2, 3]) + v * 3.8)) < 0.05


def test_debris_aimed_at_the_robot_hits_unless_it_reacts():
    r = run_hazard(with_anomalies(6001), speed=0.15)
    assert r["ignore"]["debris_collisions"] > 0
    avoid = r["avoid"]
    assert avoid["debris_collisions"] == 0 and avoid["collisions"] == 0
    assert avoid["min_debris_clearance_m"] >= 0.45
    assert avoid["success"] and avoid["targets_inspected"] == len(with_anomalies(6001).route)
    kinds = [e["event"] for e in avoid["events"]]
    assert kinds[0] == "hazard_detected" and "replanned" in kinds


def test_urgent_short_range_encounter_is_dodged():
    """Seed 6005 with a 1 m sensor: debris at a robot holding still at a viewpoint, ~4 s of warning.
    Before urgent mode the robot paused to look and got hit."""
    r = run_hazard(with_anomalies(6005), speed=0.15, sensor_range=1.0)
    assert r["ignore"]["debris_collisions"] > 0
    assert r["avoid"]["debris_collisions"] == 0
    detected = next(e for e in r["avoid"]["events"] if e["event"] == "hazard_detected")
    assert detected["urgent"]
