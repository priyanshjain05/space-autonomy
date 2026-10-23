"""How does anomaly detection degrade as reality drifts from the robot's assumptions?

Sweeps one factor at a time, others held at zero:
    navigation position error, navigation attitude error, sun-direction error,
and runs every condition twice: with and without registration (aligning the
digital twin to the camera before comparing). Every level and both pipelines
see the same randomised spacecraft (same seeds), so curves differ only by
the factor being swept and the pipeline.
    python scripts/noise_sweep.py                       # 40 missions per level, ~10 min on 11 cores
    python scripts/noise_sweep.py --runs 10             # quick look
    python scripts/noise_sweep.py --pipelines raw       # just one pipeline
    python scripts/noise_sweep.py --replot              # redraw from the last run's sweep.csv
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

from space_autonomy.evaluation.batch import aggregate, run_many, wilson
from space_autonomy.scenarios.alpha03 import nominal, with_anomalies
from space_autonomy.scenarios.conditions import Conditions
from space_autonomy.sim.anomalies import KINDS

FACTORS = {
    # key: (panel title, x-axis label, levels, level -> Conditions)
    "nav_pos": ("Navigation position error", "σ per axis (cm)", [0, 0.25, 0.5, 1, 2, 4, 8],
                lambda v: Conditions(nav_pos_sigma=v / 100)),
    "nav_att": ("Navigation attitude error", "σ per axis (deg)", [0, 0.05, 0.1, 0.25, 0.5, 1, 2],
                lambda v: Conditions(nav_att_sigma_deg=v)),
    "sun": ("Sun direction error", "angle (deg)", [0, 2, 5, 10, 20, 30],
            lambda v: Conditions(sun_error_deg=v)),
}
PIPELINES = {"registered": True, "raw": False}

# Validated categorical palette (light surface), fixed order; marker shape is the secondary encoding.
SERIES = [("registered", "With registration", "#2a78d6", "o"),
          ("raw", "Without registration", "#eb6834", "^")]
METRICS = [("false_alarm_rate", "False alarm rate (%)"),
           ("correct_rate", "Faults found with the correct type (%)")]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e0", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40, help="randomised missions per level")
    ap.add_argument("--first-seed", type=int, default=3000)
    ap.add_argument("--pipelines", default="registered,raw")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/noise_sweep")
    ap.add_argument("--replot", action="store_true", help="skip the missions; redraw from sweep.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.replot:
        with open(out / "sweep.csv") as fh:
            rows = [{k: v if k in ("factor", "pipeline") else float(v) for k, v in r.items()}
                    for r in csv.DictReader(fh)]
        plot(rows, out / "registration_comparison.png", int(rows[0]["missions"]) - 1)
        print(f"Redrew {out / 'registration_comparison.png'}")
        return

    base = [nominal()] + [with_anomalies(args.first_seed + i) for i in range(args.runs)]
    n_targets = len(nominal().route)
    # The zero level is the same for every factor, so run it once.
    conds = [("ideal", 0, Conditions())]
    conds += [(f, v, make(v)) for f, (_, _, levels, make) in FACTORS.items() for v in levels if v != 0]
    scenarios = [replace(sc, conditions=c) for _, _, c in conds for sc in base]

    results = {}
    for pipe in args.pipelines.split(","):
        t0 = time.time()
        print(f"{pipe}: {len(conds)} conditions x {len(base)} missions = {len(scenarios)} missions")
        summaries = run_many(scenarios, args.workers, lambda i, n: print(f"\r{i}/{n}", end="", flush=True),
                             registration=PIPELINES[pipe])
        print(f"  ({time.time() - t0:.0f} s)")
        for j, (f, v, _) in enumerate(conds):
            results[(pipe, f, v)] = aggregate(summaries[j * len(base):(j + 1) * len(base)], n_targets)

    rows = []
    for pipe in args.pipelines.split(","):
        for f, (_, _, levels, _) in FACTORS.items():
            for v in levels:
                r = results[(pipe, "ideal", 0) if v == 0 else (pipe, f, v)]
                lo, hi = wilson(r["type_correct"], r["injected"])
                rows.append({
                    "pipeline": pipe, "factor": f, "level": v,
                    **{k: r[k] for k in ("missions", "injected", "found", "type_correct", "healthy_checks",
                                         "false_alarms", "with_collision")},
                    **{f"fa_{k}": r["false_alarm_kinds"].get(k, 0) for k in KINDS},
                    **{f"{k}{s}": round(r[f"{k}_ci"][i] if s else r[k], 4)
                       for k in ("recall", "type_accuracy", "false_alarm_rate")
                       for i, s in ((None, ""), (0, "_lo"), (1, "_hi"))},
                    "correct_rate": round(r["type_correct"] / r["injected"], 4),
                    "correct_rate_lo": round(lo, 4), "correct_rate_hi": round(hi, 4),
                    **{k: round(r.get(k, np.nan), 5) for k in ("median_est_pos_m", "median_est_att_deg",
                                                                "median_reg_pos_m", "median_reg_att_deg",
                                                                "icp_accept_rate", "image_accept_rate")},
                })
    with open(out / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plot(rows, out / "registration_comparison.png", args.runs)
    md = table(rows, args)
    (out / "summary.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nOutputs in {out}/")


def plot(rows, path, runs):
    pipes = [s for s in SERIES if any(r["pipeline"] == s[0] for r in rows)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True, sharex="col")
    fig.patch.set_facecolor(SURFACE)
    for col, (f, (title, xlabel, levels, _)) in enumerate(FACTORS.items()):
        x = np.arange(len(levels))  # levels are roughly geometric; even spacing keeps each one readable
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for pipe, label, color, marker in pipes:
                rs = sorted((r for r in rows if r["pipeline"] == pipe and r["factor"] == f), key=lambda r: r["level"])
                y = np.array([r[metric] for r in rs]) * 100
                ax.fill_between(x, np.array([r[f"{metric}_lo"] for r in rs]) * 100,
                                np.array([r[f"{metric}_hi"] for r in rs]) * 100, color=color, alpha=0.14, lw=0)
                ax.plot(x, y, color=color, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label)
            ax.set_xticks(x, [f"{v:g}" for v in levels])
            ax.set_ylim(-3, 103)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if row == 0:
                ax.set_title(title, color=INK, fontsize=12, loc="left")
            else:
                ax.set_xlabel(xlabel, color=MUTED)
            if col == 0:
                ax.set_ylabel(ylabel, color=MUTED)
    # Direct labels on the top-left panel, in ink, where the lines are furthest apart.
    ax = axes[0, 0]
    if len(pipes) > 1:
        vals = {p: [r["false_alarm_rate"] * 100 for r in sorted((r for r in rows if r["pipeline"] == p
                                                                 and r["factor"] == "nav_pos"), key=lambda r: r["level"])]
                for p, *_ in pipes}
        at = int(np.argmax(np.abs(np.subtract(vals[pipes[0][0]], vals[pipes[1][0]]))))
        for p, label, *_ in pipes:
            # The upper line rises from the lower left, so its label goes up-left; the lower one down-right.
            above = vals[p][at] >= max(v[at] for v in vals.values())
            ax.annotate(label, (at, vals[p][at]), xytext=(-10, 6) if above else (10, -14),
                        textcoords="offset points", ha="right" if above else "left",
                        fontsize=9, color=INK, bbox=dict(fc=SURFACE, ec="none", pad=1.5))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False,
               fontsize=10, labelcolor=INK)
    fig.suptitle("Registering the digital twin before comparing", x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.005, f"{runs} randomised missions + 1 clean per level; identical spacecraft at every level and "
             "for both pipelines. Shaded bands: 95% confidence intervals (Wilson).", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def table(rows, args):
    pipes = list(dict.fromkeys(r["pipeline"] for r in rows))
    lines = [f"# Noise sweep: {args.runs} randomised missions + 1 clean per level (seeds {args.first_seed}+)", ""]
    for f, (title, xlabel, levels, _) in FACTORS.items():
        head = " | ".join(f"False alarms ({p}) | Correct ({p})" for p in pipes)
        lines += [f"## {title}", "", f"| {xlabel} | {head} | Attitude error: nav -> registered | Collisions |",
                  "|---|" + "---|---|" * len(pipes) + "---|---|"]
        for v in levels:
            by = {p: next(r for r in rows if r["pipeline"] == p and r["factor"] == f and r["level"] == v) for p in pipes}
            cells = " | ".join(f"{100 * by[p]['false_alarm_rate']:.0f}% | {100 * by[p]['correct_rate']:.0f}%" for p in pipes)
            reg = by.get("registered")
            att = (f"{reg['median_est_att_deg']:.3f} -> {reg['median_reg_att_deg']:.3f} deg"
                   if reg and np.isfinite(reg["median_reg_att_deg"]) else "-")
            lines.append(f"| {v:g} | {cells} | {att} | {sum(by[p]['with_collision'] for p in pipes):g} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
