"""What did looking again actually do? Reads the active pipeline's per-inspection log.

For every component the robot looked at more than once:
    rescued   first look wrong, final belief right
    spoiled   first look right, final belief wrong
    unchanged both right or both wrong
and whether confidence is still calibrated on the re-look viewpoints, which
the confidence model never saw in training (it was fitted on standard views).
    python scripts/relook_analysis.py results/m7_sweep/assessments_active.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

from space_autonomy.perception.calibration import auroc, expected_calibration_error

TEXT = {"verdict", "truth", "component", "mission"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/m7_sweep/assessments_active.csv")
    args = ap.parse_args()
    with open(args.path) as fh:
        rows = [{k: v if k in TEXT else float(v) for k, v in r.items()} for r in csv.DictReader(fh)]

    groups = defaultdict(list)
    for r in rows:
        key = (r["mission"], r["nav_pos_sigma"], r["nav_att_sigma_deg"], r["sun_error_deg"], r["component"])
        groups[key].append(r)

    outcome = defaultdict(int)
    looks_hist = defaultdict(int)
    for looks in groups.values():
        looks.sort(key=lambda r: r["look"])
        looks_hist[len(looks)] += 1
        if len(looks) == 1:
            continue
        first = looks[0]["correct"]
        final = max(looks, key=lambda r: r["confidence"])["correct"]
        outcome[("rescued" if final else "both wrong") if not first else ("unchanged" if final else "spoiled")] += 1

    first_looks = [r for r in rows if r["look"] == 1]
    relooks = [r for r in rows if r["look"] > 1]
    lines = ["# Re-inspection: what looking again did", "",
             f"{len(groups)} component inspections; looks per component: "
             + ", ".join(f"{k} look{'s' if k > 1 else ''}: {v}" for k, v in sorted(looks_hist.items())), "",
             "| Components looked at more than once | Count |", "|---|---|"]
    for k in ("rescued", "unchanged", "both wrong", "spoiled"):
        lines.append(f"| {k} | {outcome[k]} |")
    lines += ["", "| Calibration | First looks (standard viewpoint) | Re-looks (tilted viewpoint) |", "|---|---|---|"]
    for name, fn in (("Inspections", len), ("Accuracy", lambda rs: f"{np.mean([r['correct'] for r in rs]):.3f}"),
                     ("Mean confidence", lambda rs: f"{np.mean([r['confidence'] for r in rs]):.3f}"),
                     ("Calibration error", lambda rs: f"{expected_calibration_error([r['confidence'] for r in rs], [r['correct'] for r in rs]):.3f}"),
                     ("AUROC", lambda rs: f"{auroc([r['confidence'] for r in rs], [r['correct'] for r in rs]):.3f}")):
        lines.append(f"| {name} | {fn(first_looks)} | {fn(relooks) if relooks else '-'} |")
    report = "\n".join(lines)
    print(report)


if __name__ == "__main__":
    main()
