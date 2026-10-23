import numpy as np

from space_autonomy.autonomy.agent import make_inspection_agent
from space_autonomy.core.geometry import clearance
from space_autonomy.evaluation.anomaly_scoring import score_anomalies
from space_autonomy.evaluation.harness import run_mission
from space_autonomy.planning.inspection_planner import fixed_route_inspector
from space_autonomy.planning.paths import segment_clear
from space_autonomy.scenarios.alpha03 import nominal, showcase
from space_autonomy.sim.sensors import OnboardCamera
from space_autonomy.sim.world import SpaceSim


def test_viewpoints_are_clear_of_the_spacecraft():
    sc = nominal()
    obstacles = sc.spacecraft.obstacles()
    for t in sc.spacecraft.targets():
        assert clearance(t.viewpoint, obstacles) > 1.0, t.name


def test_planned_route_never_cuts_through_the_spacecraft():
    sc = nominal()
    insp = fixed_route_inspector(sc.vehicle, sc.spacecraft.targets(), sc.spacecraft.obstacles(),
                                 sc.home, order=sc.route)
    points = [sc.home] + [p for leg in insp.legs for p in leg.waypoints]
    for a, b in zip(points[:-1], points[1:]):
        assert segment_clear(a, b, sc.spacecraft.obstacles(), sc.vehicle.bounding_radius)


def test_baseline_mission_succeeds():
    sc = nominal()
    sim = SpaceSim(sc.vehicle, sc.spacecraft, start_pos=sc.home)
    insp = fixed_route_inspector(sc.vehicle, sc.spacecraft.targets(), sc.spacecraft.obstacles(),
                                 sc.home, order=sc.route)
    result = run_mission(sim, insp, max_time=sc.max_time, expected_targets=sc.route)
    s = result.summary
    assert s["success"], s
    assert s["collisions"] == 0
    assert s["targets_inspected"] == len(sc.route)
    assert s["min_clearance_m"] > 0.2
    assert s["final_distance_from_start_m"] < 0.1
    assert s["propellant_used_pct"] < 80
    assert np.isfinite([r["x"] for r in result.telemetry]).all()


def test_agent_finds_every_anomaly_in_showcase():
    sc = showcase()
    sim = SpaceSim(sc.vehicle, sc.world, start_pos=sc.home)
    agent = make_inspection_agent(sc.vehicle, sc.spacecraft, sc.home, sc.route)
    result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route,
                         camera=OnboardCamera(sim, seed=0), log_hz=2.0)
    s = {**result.summary, **score_anomalies(agent.world_model, sc.anomalies, sc.route)}
    assert s["success"] and s["collisions"] == 0
    assert s["anomalies_found"] == s["anomaly_type_correct"] == 4
    assert s["false_alarms"] == 0
    assert sum(e["event"] == "assessed" for e in result.events) == len(sc.route)
