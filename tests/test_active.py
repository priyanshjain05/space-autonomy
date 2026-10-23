from dataclasses import replace

import numpy as np
import pytest

from space_autonomy.autonomy.agent import make_inspection_agent
from space_autonomy.core.geometry import clearance
from space_autonomy.core.types import VehicleSpec
from space_autonomy.evaluation.anomaly_scoring import score_anomalies
from space_autonomy.evaluation.harness import run_mission
from space_autonomy.planning.inspection_planner import InspectionPlanner, vehicle_margin
from space_autonomy.planning.paths import segment_clear
from space_autonomy.planning.sequencer import Leg, LegSequencer
from space_autonomy.scenarios.alpha03 import ALPHA03, showcase
from space_autonomy.scenarios.conditions import Conditions
from space_autonomy.sim.sensors import OnboardCamera
from space_autonomy.sim.world import SpaceSim

VEHICLE = VehicleSpec()
PLANNER = InspectionPlanner(ALPHA03.obstacles(), vehicle_margin(VEHICLE))


@pytest.mark.parametrize("target", ALPHA03.targets(), ids=lambda t: t.name)
def test_alternate_viewpoints_are_safe_and_different(target):
    alts = PLANNER.alternate_viewpoints(target)
    assert alts
    for vp in alts:
        assert clearance(vp, ALPHA03.obstacles()) > PLANNER.margin
        assert np.isclose(np.linalg.norm(vp - target.point), target.standoff)
        cos = (vp - target.point) @ target.normal / target.standoff
        assert np.isclose(np.degrees(np.arccos(cos)), 30.0)
    assert PLANNER.nearest_viewpoint(target, target.viewpoint, avoid=alts) is None


def test_detour_replans_the_next_leg_from_the_detour():
    targets = {t.name: t for t in ALPHA03.targets()}
    home = np.array([-6.0, -3.0, 1.5])
    legs = PLANNER.fixed_route(ALPHA03.targets(), home, ["antenna", "thruster_1"])
    seq = LegSequencer(legs, None)
    t = targets["antenna"]
    vp = PLANNER.nearest_viewpoint(t, t.viewpoint)
    seq.detour(PLANNER.inspect_leg(t, vp, t.viewpoint, kind="reinspect"),
               lambda old, start: replace(old, waypoints=PLANNER.path(start, old.waypoints[-1])))
    assert [l.kind for l in seq.legs] == ["inspect", "reinspect", "inspect", "return"]
    # The leg after the detour must be flyable from the detour's end, not from the original viewpoint.
    points = [vp] + seq.legs[2].waypoints
    for a, b in zip(points[:-1], points[1:]):
        assert segment_clear(a, b, ALPHA03.obstacles(), VEHICLE.bounding_radius)


def run(active, att_deg=1.5):
    sc = replace(showcase(), conditions=Conditions(nav_att_sigma_deg=att_deg))
    sim = SpaceSim(sc.vehicle, sc.world, start_pos=sc.home)
    agent = make_inspection_agent(sc.vehicle, sc.spacecraft, sc.home, sc.route, active=active)
    result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route,
                         camera=OnboardCamera(sim, seed=sc.seed), nav=sc.conditions.navigation(sc.seed), log_hz=2.0)
    return result, score_anomalies(agent.world_model, sc.anomalies, sc.route), agent


def test_looking_again_fixes_a_doubtful_verdict():
    """Showcase with 1.5 deg of attitude error: the antenna's debris first reads as 'misaligned' at 13%."""
    passive, p_score, _ = run(active=False)
    active, a_score, agent = run(active=True)
    assert p_score["anomaly_type_correct"] == 3
    assert a_score["anomaly_type_correct"] == 4 and a_score["false_alarms"] == 0
    relooks = [e for e in active.events if e["event"] == "reinspect_planned"]
    assert [e["leg"] for e in relooks] == ["antenna"]
    assert agent.world_model["antenna"].observations == 2
    assert active.summary["collisions"] == 0 and active.summary["success"]
    assert active.summary["mission_time_s"] > passive.summary["mission_time_s"]  # looking again costs time


def test_confident_verdicts_are_not_rechecked():
    result, score, _ = run(active=True, att_deg=0.0)
    assert not any(e["event"] == "reinspect_planned" for e in result.events)
    assert score["anomaly_type_correct"] == 4
