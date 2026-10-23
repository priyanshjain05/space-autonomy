"""Run many missions (in parallel) and boil them down to rates with confidence intervals."""

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

from ..autonomy.agent import make_inspection_agent
from ..sim.sensors import OnboardCamera
from ..sim.world import SpaceSim
from .anomaly_scoring import score_anomalies
from .harness import run_mission


def run_scenario(sc) -> dict:
    """One full mission under the scenario's conditions -> flat summary dict."""
    c = sc.conditions
    sim = SpaceSim(sc.vehicle, sc.world, start_pos=sc.home, sun_dir=c.sun_dir(sc.seed))
    agent = make_inspection_agent(sc.vehicle, sc.spacecraft, sc.home, sc.route)
    result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route,
                         camera=OnboardCamera(sim, seed=sc.seed), nav=c.navigation(sc.seed),
                         log_hz=2.0, name=sc.name)
    s = result.summary
    s.update(score_anomalies(agent.world_model, sc.anomalies, sc.route))
    s.update(asdict(c))
    s["seed"] = sc.seed
    s["false_alarm_kinds"] = [b.kind for n, b in agent.world_model.beliefs.items()
                              if b.kind and n not in {a.component for a in sc.anomalies}]
    return s


def run_many(scenarios, workers=None, progress=None) -> list[dict]:
    """Missions in parallel processes, results in input order."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers == 1:
        out = []
        for i, sc in enumerate(scenarios):
            out.append(run_scenario(sc))
            if progress:
                progress(i + 1, len(scenarios))
        return out
    # "spawn" gives each worker a clean process, so each gets its own rendering context.
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_scenario, sc) for sc in scenarios]
        out = []
        for i, f in enumerate(futures):
            out.append(f.result())
            if progress:
                progress(i + 1, len(scenarios))
        return out


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
        "mean_propellant_pct": float(np.mean([s["propellant_used_pct"] for s in summaries])),
        "min_clearance_m": float(min(s["min_clearance_m"] for s in summaries)),
        "false_alarm_kinds": Counter(k for s in summaries for k in s["false_alarm_kinds"]),
    }
