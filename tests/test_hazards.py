import numpy as np

from space_autonomy.autonomy.hazards import Tracker, min_gap, timeline
from space_autonomy.core.types import Detection
from space_autonomy.evaluation.batch import run_hazard
from space_autonomy.scenarios.alpha03 import with_anomalies


def test_timeline_flies_waypoints_and_pauses():
    at, end = timeline([0, 0, 0], [[2, 0, 0], [2, 2, 0]], t0=0.0, speed=1.0, wait=1.0, pauses={0: 3.0})
    assert np.allclose(at(0.5), [0, 0, 0])      # still waiting
    assert np.allclose(at(2.0), [1, 0, 0])      # halfway along the first segment
    assert np.allclose(at(4.5), [2, 0, 0])      # pausing at the first waypoint
    assert np.allclose(at(7.0), [2, 1, 0])      # then on to the next
    assert np.isclose(end, 8.0)


def test_min_gap_head_on():
    at, _ = timeline([0, 0, 0], [], 0.0, 1.0)
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
