"""How does anomaly detection degrade as reality drifts from the robot's assumptions?

Sweeps one factor at a time, others held at zero:
    navigation position error, navigation attitude error, sun-direction error.
Every level reuses the same randomised spacecraft (same seeds), so the curves
differ only by the factor being swept.
    python scripts/noise_sweep.py                 # 40 missions per level, ~4 min on 8+ cores
    python scripts/noise_sweep.py --runs 10       # quick look
    python scripts/noise_sweep.py --replot        # redraw from the last run's sweep.csv
"""

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from space_autonomy.evaluation.batch import aggregate, run_many
from space_autonomy.scenarios.alpha03 import nominal, with_anomalies
from space_autonomy.scenarios.conditions import Conditions
from space_autonomy.sim.anomalies import KINDS

FACTORS = {
    # key: (panel title, x-axis label, levels, level -> Conditions)
    "nav_pos": ("Navigation position error", "σ per axis (cm)", [0, 0.25, 0.5, 1, 2, 4],
                lambda v: Conditions(nav_pos_sigma=v / 100)),
    "nav_att": ("Navigation attitude error", "σ per axis (deg)", [0, 0.05, 0.1, 0.25, 0.5, 1],
                lambda v: Conditions(nav_att_sigma_deg=v)),
    "sun": ("Sun direction error", "angle (deg)", [0, 1, 2, 5, 10, 20],
            lambda v: Conditions(sun_error_deg=v)),
}

# Validated categorical palette (light surface), fixed order; marker shape is the secondary encoding.
SERIES = [
    ("recall", "Anomalies found", "#2a78d6", "o"),
    ("type_accuracy", "Correct type (of found)", "#1baf7a", "s"),
    ("false_alarm_rate", "False alarm rate", "#eb6834", "^"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40, help="randomised missions per level")
    ap.add_argument("--first-seed", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/noise_sweep")
    ap.add_argument("--replot", action="store_true", help="skip the missions; redraw from sweep.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.replot:
        with open(out / "sweep.csv") as fh:
            rows = [{k: v if k == "factor" else float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        args.runs = int(rows[0]["missions"]) - 1
        plot(rows, out / "degradation.png", args.runs)
        print(f"Redrew {out / 'degradation.png'}")
        return

    base = [nominal()] + [with_anomalies(args.first_seed + i) for i in range(args.runs)]
    n_targets = len(nominal().route)

    # The zero level is the same for every factor, so run it once.
    jobs = [("ideal", 0, Conditions())]
    jobs += [(f, v, make(v)) for f, (_, _, levels, make) in FACTORS.items() for v in levels if v != 0]
    scenarios = [replace(sc, conditions=c) for _, _, c in jobs for sc in base]

    t0 = time.time()
    print(f"{len(jobs)} conditions x {len(base)} missions = {len(scenarios)} missions")
    summaries = run_many(scenarios, args.workers, lambda i, n: print(f"\r{i}/{n}", end="", flush=True))
    print(f"  ({time.time() - t0:.0f} s)")

    results = {}
    for j, (f, v, c) in enumerate(jobs):
        results[(f, v)] = aggregate(summaries[j * len(base):(j + 1) * len(base)], n_targets)

    rows = []
    for f, (title, _, levels, _) in FACTORS.items():
        for v in levels:
            r = results[("ideal", 0) if v == 0 else (f, v)]
            rows.append({"factor": f, "level": v, **{k: r[k] for k in (
                "missions", "injected", "found", "type_correct", "healthy_checks", "false_alarms",
                "with_collision")},
                **{f"fa_{k}": r["false_alarm_kinds"].get(k, 0) for k in KINDS},
                **{f"{k}{s}": round(r[f"{k}_ci"][i] if s else r[k], 4)
                   for k in ("recall", "type_accuracy", "false_alarm_rate")
                   for i, s in ((None, ""), (0, "_lo"), (1, "_hi"))}})
    with open(out / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plot(rows, out / "degradation.png", args.runs)
    md = table(rows, args)
    (out / "summary.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nOutputs in {out}/")


def plot(rows, path, runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (f, (title, xlabel, levels, _)) in zip(axes, FACTORS.items()):
        ax.set_facecolor("#fcfcfb")
        rs = [r for r in rows if r["factor"] == f]
        x = np.arange(len(levels))  # levels are roughly geometric; even spacing keeps each one readable
        for key, label, color, marker in SERIES:
            y = np.array([r[key] for r in rs]) * 100
            lo = np.array([r[f"{key}_lo"] for r in rs]) * 100
            hi = np.array([r[f"{key}_hi"] for r in rs]) * 100
            ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
            ax.plot(x, y, color=color, lw=2, marker=marker, ms=7, mec="#fcfcfb", mew=1.5, label=label)
        ax.set_xticks(x, [f"{v:g}" for v in levels])
        ax.set_xlabel(xlabel, color=MUTED)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("rate (%)", color=MUTED)
    # Direct labels on the first panel, in ink (the marker carries the colour), at the level where
    # the three lines are furthest apart so the labels can't collide.
    first = [r for r in rows if r["factor"] == "nav_pos"]
    vals = np.array([[r[key] for key, *_ in SERIES] for r in first]) * 100
    gaps = [min(abs(a - b) for i, a in enumerate(v) for b in v[i + 1:]) for v in vals]
    at = int(np.argmax(gaps))
    for (key, label, color, marker), y in zip(SERIES, vals[at]):
        below = y == vals[at].min()
        axes[0].annotate(label, (at, y), xytext=(8, -14 if below else 8), textcoords="offset points",
                         ha="left", fontsize=9, color=INK, bbox=dict(fc="#fcfcfb", ec="none", pad=1.5))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False,
               fontsize=10, labelcolor=INK)
    fig.suptitle("How model-based anomaly detection degrades", x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.005, f"{runs} randomised missions + 1 clean per level, same spacecraft at every level. "
             "Shaded bands: 95% confidence intervals (Wilson).", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def table(rows, args):
    lines = [f"# Noise sweep: {args.runs} randomised missions + 1 clean per level (seeds {args.first_seed}+)", ""]
    for f, (title, xlabel, _, _) in FACTORS.items():
        lines += [f"## {title}", "", f"| {xlabel} | Found | Correct type | False alarms | ...as which verdict | Collisions |",
                  "|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["factor"] == f):
            lines.append(f"| {r['level']:g} | {100 * r['recall']:.0f}% | {100 * r['type_accuracy']:.0f}% | "
                         f"{100 * r['false_alarm_rate']:.0f}% ({r['false_alarms']}/{r['healthy_checks']}) | "
                         f"{', '.join(f'{k} {r[f'fa_{k}']}' for k in KINDS if r[f'fa_{k}']) or '-'} | "
                         f"{r['with_collision']} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
