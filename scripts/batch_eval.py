"""Run many randomised missions and summarise how the autonomy did.

Each seed gives a different set of 1-3 hidden anomalies. One mission is
an anecdote; this turns it into numbers.
    python scripts/batch_eval.py --runs 100
    python scripts/batch_eval.py --runs 100 --nav-pos-cm 1 --nav-att-deg 0.2 --sun-deg 5
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from space_autonomy.evaluation.batch import aggregate, run_many
from space_autonomy.scenarios.alpha03 import nominal, with_anomalies
from space_autonomy.scenarios.conditions import Conditions
from space_autonomy.sim.anomalies import KINDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=2000)  # 1000-1099 were used while developing M6
    ap.add_argument("--nav-pos-cm", type=float, default=0.0, help="navigation position error, cm per axis")
    ap.add_argument("--nav-att-deg", type=float, default=0.0, help="navigation attitude error, deg per axis")
    ap.add_argument("--sun-deg", type=float, default=0.0, help="error in the sun direction the twin assumes")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/batch")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cond = Conditions(args.nav_pos_cm / 100, args.nav_att_deg, args.sun_deg)
    # One clean spacecraft too: anything flagged there is a false alarm.
    scenarios = [replace(sc, conditions=cond) for sc in
                 [nominal()] + [with_anomalies(args.first_seed + i) for i in range(args.runs)]]
    t0 = time.time()
    summaries = run_many(scenarios, args.workers,
                         lambda i, n: print(f"\r{i}/{n} missions", end="", flush=True))
    print(f"  ({time.time() - t0:.0f} s)\n")

    agg = aggregate(summaries, len(nominal().route))
    per_kind = defaultdict(lambda: [0, 0, 0])  # injected, found, correct type
    for s in summaries:
        for d in s["anomaly_details"]:
            k = per_kind[d["injected"]]
            k[0] += 1
            k[1] += d["reported"] in KINDS
            k[2] += d["reported"] == d["injected"]

    pct = lambda r, ci: f"{100 * r:.0f}% (95% CI {100 * ci[0]:.0f}-{100 * ci[1]:.0f}%)"
    lines = [
        "# Batch evaluation: fixed-route baseline + model-based change detection",
        "",
        f"{agg['missions']} missions ({args.runs} randomised + 1 clean), seeds "
        f"{args.first_seed}-{args.first_seed + args.runs - 1}, conditions: {cond.label()}",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Missions completed | {agg['completed']} / {agg['missions']} |",
        f"| Missions with a collision | {agg['with_collision']} / {agg['missions']} |",
        f"| Anomalies found | {agg['found']} / {agg['injected']}: {pct(agg['recall'], agg['recall_ci'])} |",
        f"| ... with the correct type | {agg['type_correct']} / {agg['found']} |",
        f"| False alarms | {agg['false_alarms']} in {agg['healthy_checks']} healthy-component checks: "
        f"{pct(agg['false_alarm_rate'], agg['false_alarm_rate_ci'])} |",
        f"| Mean mission time | {agg['mean_time_s']:.0f} s |",
        f"| Mean propellant used | {agg['mean_propellant_pct']:.1f}% |",
        f"| Min clearance (worst mission) | {agg['min_clearance_m']:.2f} m |",
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
