"""Run many missions (in parallel) and boil them down to rates with confidence intervals."""

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, replace
from multiprocessing import get_context

import numpy as np

from ..autonomy.agent import make_inspection_agent
from ..core.rotations import quat_angle
from ..sim.hazards import DebrisInjector, HazardSensor, plan_strike
from ..sim.sensors import OnboardCamera
from ..sim.world import SpaceSim
from .anomaly_scoring import score_anomalies
from .harness import run_mission


def run_scenario(sc, registration=True, calibrated=True, active=True) -> dict:
    """One full mission under the scenario's conditions -> flat summary dict."""
    c = sc.conditions
    sim = SpaceSim(sc.vehicle, sc.world, start_pos=sc.home, sun_dir=c.sun_dir(sc.seed))
    agent = make_inspection_agent(sc.vehicle, sc.spacecraft, sc.home, sc.route, registration=registration,
                                  calibrated=calibrated, active=active)
    result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route,
                         camera=OnboardCamera(sim, seed=sc.seed), nav=c.navigation(sc.seed),
                         log_hz=2.0, name=sc.name)
    s = result.summary
    s.update(score_anomalies(agent.world_model, sc.anomalies, sc.route))
    s.update(asdict(c))
    s["seed"] = sc.seed
    s["registration"] = registration
    s["active"] = agent.active is not None
    s["reinspections"] = sum(e["event"] == "reinspect_planned" for e in result.events)
    s["pose_errors"] = pose_errors(result.events)
    s["assessments"] = assessment_rows(result.events, sc)
    s["false_alarm_kinds"] = [b.kind for n, b in agent.world_model.beliefs.items()
                              if b.kind and n not in {a.component for a in sc.anomalies}]
    return s


def assessment_rows(events, sc) -> list[dict]:
    """One row per inspection: what the robot observed (features), said, and whether it was right."""
    truth = {a.component: a.kind for a in sc.anomalies}
    rows = []
    for e in events:
        if e["event"] != "assessed":
            continue
        actual = truth.get(e["leg"], "nominal")
        rows.append({"mission": sc.name, "seed": sc.seed, **asdict(sc.conditions), "component": e["leg"],
                     "look": e.get("look", 1),
                     "truth": actual, "verdict": e["result"], "correct": int(e["result"] == actual),
                     "heuristic_confidence": e["heuristic_confidence"], "confidence": e["confidence"],
                     **e["features"]})
    return rows


def pose_errors(events) -> list[dict]:
    """Per inspection: how far the navigation estimate (and the registered pose) were from the truth."""
    out = []
    for e in events:
        if e["event"] != "assessed" or "true_position" not in e:
            continue
        tp, tq = np.array(e["true_position"]), np.array(e["true_quat"])
        row = {"est_pos_m": float(np.linalg.norm(np.array(e["est_position"]) - tp)),
               "est_att_deg": float(np.degrees(quat_angle(e["est_quat"], tq)))}
        if "refined_position" in e:
            row["reg_pos_m"] = float(np.linalg.norm(np.array(e["refined_position"]) - tp))
            row["reg_att_deg"] = float(np.degrees(quat_angle(e["refined_quat"], tq)))
            row["icp_accepted"] = e["icp_accepted"]
            row["image_accepted"] = e["image_accepted"]
        out.append(row)
    return out


def run_hazard(sc, speed=0.15, radius=0.25, sensor_range=4.0, hazard_config=None) -> dict:
    """One debris encounter, flown three ways on the same spacecraft:

    clean   no debris; its trajectory is what the adversary aims at
    ignore  debris launched to hit the robot; the robot doesn't react to it
    avoid   same debris; the robot detects it, stops, replans
    """
    rng = np.random.default_rng(sc.seed + 7)
    vehicle = replace(sc.vehicle, hazard_range=sensor_range)

    def fly(avoid, strike=None):
        sim = SpaceSim(vehicle, sc.world, start_pos=sc.home, debris_radii=(radius,) if strike else ())
        agent = make_inspection_agent(vehicle, sc.spacecraft, sc.home, sc.route, avoid=avoid,
                                      **({"hazard_config": hazard_config} if hazard_config else {}))
        result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route,
                             camera=OnboardCamera(sim, seed=sc.seed),
                             hazard_sensor=HazardSensor(sim, seed=sc.seed) if strike else None,
                             injector=DebrisInjector(strike) if strike else None,
                             log_hz=10.0 if strike is None else 2.0, name=sc.name)
        s = result.summary
        s.update(score_anomalies(agent.world_model, sc.anomalies, sc.route))
        s["events"] = [e for e in result.events if e["event"] in
                       ("hazard_detected", "replanned", "resumed", "no_safe_option")]
        return s, result

    clean, clean_result = fly(avoid=False)
    strike = plan_strike(clean_result.telemetry, sc.world.obstacles(), speed, radius, rng)
    out = {"seed": sc.seed, "speed": speed, "radius": radius, "sensor_range": sensor_range,
           "aim_t": strike.aim_t, "robot_moving": strike.robot_moving, "clean": clean}
    for avoid in (False, True):
        out["avoid" if avoid else "ignore"] = fly(avoid, strike)[0]
    return out


def run_hazard_job(job) -> dict:
    """run_hazard(scenario, speed, sensor_range[, hazard_config]) as one picklable job, for run_many."""
    sc, speed, sensor_range, *config = job
    return run_hazard(sc, speed=speed, sensor_range=sensor_range, hazard_config=config[0] if config else None)


def run_many(scenarios, workers=None, progress=None, fn=None, **options) -> list[dict]:
    """Missions in parallel processes, results in input order. fn(scenario, **options) runs one."""
    fn = fn or run_scenario
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers == 1:
        out = []
        for i, sc in enumerate(scenarios):
            out.append(fn(sc, **options))
            if progress:
                progress(i + 1, len(scenarios))
        return out
    # One maths thread per worker: the parallelism is already across processes, and letting each
    # worker's linear algebra spawn its own threads too oversubscribes the CPU (~4x slower here).
    # Workers inherit the environment when they start, before they import NumPy.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    # "spawn" gives each worker a clean process, so each gets its own rendering context.
    # If a worker dies outright (a native crash takes the whole pool down with it), start a fresh pool
    # for the jobs that hadn't finished. Every job is deterministic from its seed, so a retry gives the
    # same answer it would have the first time.
    out, todo, done = [None] * len(scenarios), list(range(len(scenarios))), 0
    for attempt in range(4):
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(fn, scenarios[i], **options): i for i in todo}
            try:
                for f in as_completed(futures):
                    out[futures[f]] = f.result()
                    done += 1
                    if progress:
                        progress(done, len(scenarios))
            except BrokenProcessPool:
                todo = [i for i in todo if out[i] is None]
                print(f"\n[run_many] a worker crashed; restarting for the {len(todo)} unfinished jobs", flush=True)
                continue
        return out
    raise RuntimeError(f"run_many: workers kept crashing; {len(todo)} jobs never finished")


def wilson(k, n, z=1.96):
    """95% confidence interval for a rate k/n (Wilson score; behaves well near 0% and 100%)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(summaries, n_targets) -> dict:
    """
    recall           anomalies found / injected
    type_accuracy    correct type / found
    false_alarm_rate healthy components flagged / healthy components inspected
    """
    injected = sum(s["anomalies_injected"] for s in summaries)
    found = sum(s["anomalies_found"] for s in summaries)
    correct = sum(s["anomaly_type_correct"] for s in summaries)
    healthy = sum(n_targets - s["anomalies_injected"] for s in summaries)
    fa = sum(s["false_alarms"] for s in summaries)
    return {
        "missions": len(summaries),
        "completed": sum(s["complete"] for s in summaries),
        "with_collision": sum(s["collisions"] > 0 for s in summaries),
        "injected": injected, "found": found, "type_correct": correct,
        "healthy_checks": healthy, "false_alarms": fa,
        "recall": found / injected if injected else np.nan, "recall_ci": wilson(found, injected),
        "type_accuracy": correct / found if found else np.nan, "type_accuracy_ci": wilson(correct, found),
        "false_alarm_rate": fa / healthy if healthy else np.nan, "false_alarm_rate_ci": wilson(fa, healthy),
        "mean_time_s": float(np.mean([s["mission_time_s"] for s in summaries])),
        "time_sem_s": float(np.std([s["mission_time_s"] for s in summaries]) / np.sqrt(len(summaries))),
        "mean_propellant_pct": float(np.mean([s["propellant_used_pct"] for s in summaries])),
        "mean_reinspections": float(np.mean([s.get("reinspections", 0) for s in summaries])),
        "min_clearance_m": float(min(s["min_clearance_m"] for s in summaries)),
        "false_alarm_kinds": Counter(k for s in summaries for k in s["false_alarm_kinds"]),
        **_pose_error_medians(summaries),
    }


def _pose_error_medians(summaries) -> dict:
    rows = [r for s in summaries for r in s.get("pose_errors", [])]
    keys = ("est_pos_m", "est_att_deg", "reg_pos_m", "reg_att_deg")
    out = {f"median_{k}": float(np.median([r[k] for r in rows if k in r])) if any(k in r for r in rows)
           else np.nan for k in keys}
    if rows and "icp_accepted" in rows[0]:
        out["icp_accept_rate"] = float(np.mean([r["icp_accepted"] for r in rows]))
        out["image_accept_rate"] = float(np.mean([r["image_accepted"] for r in rows]))
    return out
