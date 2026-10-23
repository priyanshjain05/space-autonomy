"""Run many randomised missions and summarise how the autonomy did.

Each seed gives a different set of 1-3 hidden anomalies. One mission is
an anecdote; this turns it into numbers.
    python scripts/batch_eval.py --runs 50
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from space_autonomy.autonomy.agent import make_inspection_agent
from space_autonomy.evaluation.anomaly_scoring import score_anomalies
from space_autonomy.evaluation.harness import run_mission
from space_autonomy.scenarios.alpha03 import nominal, with_anomalies
from space_autonomy.sim.anomalies import KINDS
from space_autonomy.sim.sensors import OnboardCamera
from space_autonomy.sim.world import SpaceSim


def run_one(sc):
    sim = SpaceSim(sc.vehicle, sc.world, start_pos=sc.home)
    agent = make_inspection_agent(sc.vehicle, sc.spacecraft, sc.home, sc.route)
    result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route,
                         camera=OnboardCamera(sim, seed=sc.seed), log_hz=2.0, name=sc.name)
    result.summary.update(score_anomalies(agent.world_model, sc.anomalies, sc.route))
    return result.summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=1000)  # kept apart from the seeds used while developing
    ap.add_argument("--out", default="results/batch")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summaries = [run_one(nominal())]  # one clean spacecraft: any finding here is a false alarm
    for i in range(args.runs):
        summaries.append(run_one(with_anomalies(args.first_seed + i)))
        print(f"\r{i + 1}/{args.runs} missions", end="", flush=True)
    print(f"  ({time.time() - t0:.0f} s)\n")

    per_kind = defaultdict(lambda: [0, 0, 0])  # injected, found, correct type
    for s in summaries:
        for d in s["anomaly_details"]:
            k = per_kind[d["injected"]]
            k[0] += 1
            k[1] += d["reported"] in KINDS
            k[2] += d["reported"] == d["injected"]

    n = len(summaries)
    injected = sum(v[0] for v in per_kind.values())
    found = sum(v[1] for v in per_kind.values())
    correct = sum(v[2] for v in per_kind.values())
    healthy_checks = sum(len(nominal().route) - s["anomalies_injected"] for s in summaries)
    false_alarms = sum(s["false_alarms"] for s in summaries)
    lines = [
        f"# Batch evaluation: fixed-route baseline + model-based change detection",
        "",
        f"{n} missions ({args.runs} randomised + 1 clean), seeds {args.first_seed}-{args.first_seed + args.runs - 1}",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Missions completed | {sum(s['complete'] for s in summaries)} / {n} |",
        f"| Missions with a collision | {sum(s['collisions'] > 0 for s in summaries)} / {n} |",
        f"| Anomalies found | {found} / {injected} ({100 * found / injected:.0f}%) |",
        f"| ... with the correct type | {correct} / {found} |",
        f"| False alarms | {false_alarms} in {healthy_checks} healthy-component checks |",
        f"| Mean mission time | {sum(s['mission_time_s'] for s in summaries) / n:.0f} s |",
        f"| Mean propellant used | {sum(s['propellant_used_pct'] for s in summaries) / n:.1f}% |",
        f"| Min clearance (worst mission) | {min(s['min_clearance_m'] for s in summaries):.2f} m |",
        "",
        "| Anomaly type | Injected | Found | Correct type |",
        "|---|---|---|---|",
    ] + [f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in sorted(per_kind.items())]
    report = "\n".join(lines)
    print(report)

    (out / "summary.md").write_text(report + "\n")
    (out / "missions.json").write_text(json.dumps(summaries, indent=2, default=str))
    misses = [(s["mission"], d) for s in summaries for d in s["anomaly_details"] if d["reported"] != d["injected"]]
    with open(out / "misses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mission", "component", "injected", "detail", "reported", "confidence"])
        for m, d in misses:
            w.writerow([m, d["component"], d["injected"], d["detail"], d["reported"], d["confidence"]])
    if misses:
        print(f"\nMisclassified or missed ({len(misses)}): see {out / 'misses.csv'}")
        for m, d in misses[:15]:
            print(f"  {m:<20}{d['component']:<15}{d['injected']:<16}-> {d['reported']:<16}{d['detail']}")


if __name__ == "__main__":
    main()
