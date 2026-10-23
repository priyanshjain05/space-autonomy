"""How does anomaly detection degrade as reality drifts from the robot's assumptions?

Sweeps one factor at a time, others held at zero:
    navigation position error, navigation attitude error, sun-direction error,
and runs every condition through up to three pipelines:
    raw         compare against the digital twin as rendered from the pose estimate
    registered  align the twin to the camera first (M6.1)
    active      ...and look again from another viewpoint when the calibrated confidence is low (M7)
Every level and every pipeline sees the same randomised spacecraft (same seeds),
so curves differ only by the factor being swept and the pipeline.

Seeds default to 5000+: the confidence model was fitted on 3000+ and tested on
4000+, so judging re-inspection on either would flatter it.
    python scripts/noise_sweep.py                       # 40 missions per level, all three, ~40 min on 11 cores
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
PIPELINES = {"active": dict(registration=True, active=True),
             "registered": dict(registration=True, active=False),
             "raw": dict(registration=False, active=False)}

# Validated categorical palette (light surface), fixed order; marker shape is the secondary encoding.
# Colours follow the pipeline from chart to chart: registered is always blue, raw always orange.
SERIES = [("registered", "With registration", "#2a78d6", "o"),
          ("raw", "Without registration", "#eb6834", "^"),
          ("active", "Registration + re-inspection", "#1baf7a", "s")]
METRICS = [("false_alarm_rate", "False alarm rate (%)", 100),
           ("correct_rate", "Faults found with the correct type (%)", 100),
           ("mean_time_s", "Mission time (s)", 1)]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e0", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40, help="randomised missions per level")
    ap.add_argument("--first-seed", type=int, default=5000)
    ap.add_argument("--pipelines", default="active,registered,raw")
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
        plot(rows, out / "comparison.png", int(rows[0]["missions"]) - 1)
        print(f"Redrew {out / 'comparison.png'}")
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
                             **PIPELINES[pipe])
        print(f"  ({time.time() - t0:.0f} s)")
        for j, (f, v, _) in enumerate(conds):
            results[(pipe, f, v)] = aggregate(summaries[j * len(base):(j + 1) * len(base)], n_targets)
        # Every inspection, with the robot's features and the truth: training/test data for calibrate.py.
        rows_out = [r for s in summaries for r in s["assessments"]]
        with open(out / f"assessments_{pipe}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)

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
                    "mean_time_s": round(r["mean_time_s"], 2),
                    "mean_time_s_lo": round(r["mean_time_s"] - 1.96 * r["time_sem_s"], 2),
                    "mean_time_s_hi": round(r["mean_time_s"] + 1.96 * r["time_sem_s"], 2),
                    "mean_propellant_pct": round(r["mean_propellant_pct"], 2),
                    "mean_reinspections": round(r["mean_reinspections"], 3),
                    **{k: round(r.get(k, np.nan), 5) for k in ("median_est_pos_m", "median_est_att_deg",
                                                                "median_reg_pos_m", "median_reg_att_deg",
                                                                "icp_accept_rate", "image_accept_rate")},
                })
    with open(out / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plot(rows, out / "comparison.png", args.runs)
    md = table(rows, args)
    (out / "summary.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nOutputs in {out}/")


def plot(rows, path, runs):
    pipes = [p for p in SERIES if any(r["pipeline"] == p[0] for r in rows)]
    fig, axes = plt.subplots(len(METRICS), 3, figsize=(15, 3.9 * len(METRICS)), sharey="row", sharex="col")
    fig.patch.set_facecolor(SURFACE)
    for col, (f, (title, xlabel, levels, _)) in enumerate(FACTORS.items()):
        x = np.arange(len(levels))  # levels are roughly geometric; even spacing keeps each one readable
        for row, (metric, ylabel, scale) in enumerate(METRICS):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for pipe, label, color, marker in pipes:
                rs = sorted((r for r in rows if r["pipeline"] == pipe and r["factor"] == f), key=lambda r: r["level"])
                ax.fill_between(x, np.array([r[f"{metric}_lo"] for r in rs]) * scale,
                                np.array([r[f"{metric}_hi"] for r in rs]) * scale, color=color, alpha=0.14, lw=0)
                ax.plot(x, np.array([r[metric] for r in rs]) * scale, color=color, lw=2, marker=marker, ms=7,
                        mec=SURFACE, mew=1.5, label=label)
            ax.set_xticks(x, [f"{v:g}" for v in levels])
            if scale == 100:
                ax.set_ylim(-3, 103)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if row == 0:
                ax.set_title(title, color=INK, fontsize=12, loc="left")
            if row == len(METRICS) - 1:
                ax.set_xlabel(xlabel, color=MUTED)
            if col == 0:
                ax.set_ylabel(ylabel, color=MUTED)
    _direct_labels(axes[0, 0], rows, pipes)
    if any(m[0] == "mean_time_s" for m in METRICS) and len(pipes) > 2:
        axes[-1, 0].text(0.02, 0.92, "without re-inspection, both pipelines fly the same\nroute, so their lines coincide",
                         transform=axes[-1, 0].transAxes, fontsize=8, color=MUTED, va="top")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=len(pipes), frameon=False,
               fontsize=10, labelcolor=INK)
    title = ("Looking again when unsure" if any(p[0] == "active" for p in pipes)
             else "Registering the digital twin before comparing")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.005, f"{runs} randomised missions + 1 clean per level; identical spacecraft at every level and "
             "for every pipeline. Shaded bands: 95% confidence intervals.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def _direct_labels(ax, rows, pipes):
    """Label each line in the top-left panel at its last point, in ink (the marker carries the colour).

    The column is widened a little to make room, so labels sit inside the panel and away from the lines.
    """
    if len(pipes) < 2:
        return
    last = {}
    for p, label, *_ in pipes:
        rs = sorted((r for r in rows if r["pipeline"] == p and r["factor"] == "nav_pos"), key=lambda r: r["level"])
        last[p] = (len(rs) - 1, rs[-1]["false_alarm_rate"] * 100, label)
    n = max(i for i, _, _ in last.values()) + 1
    ax.set_xlim(-0.4, n - 1 + 2.6)
    ys = sorted((y, p) for p, (_, y, _) in last.items())
    placed = []
    for y, p in ys:  # nudge labels apart if two lines end close together
        y_text = max([y] + [q + 7 for q in placed])
        placed.append(y_text)
        i, _, label = last[p]
        ax.annotate(label, (i, y), xytext=(i + 0.25, y_text), textcoords="data", va="center", fontsize=9,
                    color=INK)


def table(rows, args):
    pipes = list(dict.fromkeys(r["pipeline"] for r in rows))
    lines = [f"# Noise sweep: {args.runs} randomised missions + 1 clean per level (seeds {args.first_seed}+)", ""]
    for f, (title, xlabel, levels, _) in FACTORS.items():
        head = " | ".join(f"FA {p} | Correct {p}" for p in pipes)
        extra = " | Re-looks/mission (active) | Time s (registered -> active)" if "active" in pipes else ""
        lines += [f"## {title}", "", f"| {xlabel} | {head}{extra} | Collisions |",
                  "|---|" + "---|---|" * len(pipes) + ("---|---|" if extra else "") + "---|"]
        for v in levels:
            by = {p: next(r for r in rows if r["pipeline"] == p and r["factor"] == f and r["level"] == v) for p in pipes}
            cells = " | ".join(f"{100 * by[p]['false_alarm_rate']:.0f}% | {100 * by[p]['correct_rate']:.0f}%" for p in pipes)
            if extra:
                base = by.get("registered", by["active"])
                cells += (f" | {by['active']['mean_reinspections']:.2f} | "
                          f"{base['mean_time_s']:.0f} -> {by['active']['mean_time_s']:.0f}")
            lines.append(f"| {v:g} | {cells} | {sum(by[p]['with_collision'] for p in pipes):g} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
