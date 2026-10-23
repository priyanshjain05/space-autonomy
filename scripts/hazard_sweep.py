"""Unexpected debris: does the robot get out of the way, and what does it cost?

Every mission gets one piece of debris aimed (by the adversarial injector) to
hit the robot. Each encounter is flown twice on the same spacecraft: once with
the robot ignoring loose objects, once with detect-stop-replan. Two sweeps:
    debris speed   faster debris = less warning
    sensor range   shorter range = less warning
    python scripts/hazard_sweep.py               # 40 encounters per level, ~25 min on 11 cores
    python scripts/hazard_sweep.py --runs 5      # quick look
    python scripts/hazard_sweep.py --replot
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from space_autonomy.evaluation.batch import run_hazard_job, run_many, wilson
from space_autonomy.scenarios.alpha03 import with_anomalies

DEFAULT_SPEED, DEFAULT_RANGE = 0.15, 4.0
FACTORS = {
    "speed": ("Debris speed", "m/s (sensor range 4 m)", [0.05, 0.1, 0.15, 0.2, 0.3]),
    "range": ("Hazard sensor range", "m (debris at 0.15 m/s)", [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
}
# Validated categorical palette, fixed order; marker shape is the secondary encoding.
SERIES = [("avoid", "Detects, stops and replans", "#2a78d6", "o"),
          ("ignore", "Ignores loose objects", "#eb6834", "^")]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e0", "#fcfcfb"
SAFETY = 0.5  # m, the gap the robot is told to keep (HazardConfig.safety)


def conditions():
    seen = []
    for f, (_, _, levels) in FACTORS.items():
        for v in levels:
            c = (v, DEFAULT_RANGE) if f == "speed" else (DEFAULT_SPEED, v)
            if c not in seen:
                seen.append(c)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--first-seed", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/hazard_sweep")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.replot:
        base = [with_anomalies(args.first_seed + i) for i in range(args.runs)]
        jobs = [(sc, speed, rng_m) for speed, rng_m in conditions() for sc in base]  # one pool for all of them
        t0 = time.time()
        results = run_many(jobs, args.workers, lambda i, n: print(f"\r{i}/{n} encounters", end="", flush=True),
                           fn=run_hazard_job)
        print(f"\n{len(results)} encounters x 3 missions in {time.time() - t0:.0f} s")
        (out / "encounters.json").write_text(json.dumps(results, indent=1, default=float))
    results = json.loads((out / "encounters.json").read_text())

    rows = summarise(results)
    with open(out / "hazard_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, out / "hazard_sweep.png", len(results) // len(conditions()))
    md = table(rows, results)
    (out / "summary.md").write_text(md + "\n")
    print(md)
    print(f"\nOutputs in {out}/")


def summarise(results):
    rows = []
    for f, (_, _, levels) in FACTORS.items():
        for v in levels:
            speed, rng_m = (v, DEFAULT_RANGE) if f == "speed" else (DEFAULT_SPEED, v)
            enc = [r for r in results if np.isclose(r["speed"], speed) and np.isclose(r["sensor_range"], rng_m)]
            for pipe, *_ in SERIES:
                ss = [r[pipe] for r in enc]
                hit = sum(s["debris_collisions"] > 0 for s in ss)
                gaps = np.array([s["min_debris_clearance_m"] for s in ss], float)
                extra = np.array([s["mission_time_s"] - r["clean"]["mission_time_s"] for s, r in zip(ss, enc)])
                fuel = np.array([s["propellant_used_pct"] - r["clean"]["propellant_used_pct"] for s, r in zip(ss, enc)])
                lead = [e["seconds_to_conflict"] for s in ss for e in s["events"][:1] if e["event"] == "hazard_detected"]
                lo, hi = wilson(hit, len(ss))
                rows.append({
                    "factor": f, "level": v, "pipeline": pipe, "encounters": len(ss),
                    "collision_rate": hit / len(ss), "collision_rate_lo": lo, "collision_rate_hi": hi,
                    "median_gap_m": float(np.median(gaps)), "gap_p10_m": float(np.percentile(gaps, 10)),
                    "gap_p90_m": float(np.percentile(gaps, 90)),
                    "kept_safety_rate": float(np.mean(gaps >= SAFETY - 0.05)),
                    "mean_extra_time_s": float(extra.mean()),
                    "extra_time_lo": float(extra.mean() - 1.96 * extra.std() / np.sqrt(len(extra))),
                    "extra_time_hi": float(extra.mean() + 1.96 * extra.std() / np.sqrt(len(extra))),
                    "mean_extra_propellant_pct": float(fuel.mean()),
                    "success_rate": float(np.mean([s["success"] for s in ss])),
                    "median_warning_s": float(np.median(lead)) if lead else np.nan,
                    "no_safe_option_rate": float(np.mean([s["hazard_no_safe_option"] > 0 for s in ss])),
                    "false_alarms": int(sum(s["false_alarms"] for s in ss)),
                    "type_correct": int(sum(s["anomaly_type_correct"] for s in ss)),
                    "injected": int(sum(s["anomalies_injected"] for s in ss)),
                })
    return rows


METRICS = [("collision_rate", "Missions where debris hit\nthe robot (%)", 100, "collision_rate_lo", "collision_rate_hi"),
           ("median_gap_m", "Closest approach (m)\nmedian, 10th–90th pct", 1, "gap_p10_m", "gap_p90_m"),
           ("mean_extra_time_s", "Extra mission time vs.\nno debris (s)", 1, "extra_time_lo", "extra_time_hi")]


def plot(rows, path, runs):
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(12, 10.5), sharex="col", sharey="row")
    fig.patch.set_facecolor(SURFACE)
    for col, (f, (title, xlabel, levels)) in enumerate(FACTORS.items()):
        x = np.arange(len(levels))
        for row, (metric, ylabel, scale, lo_k, hi_k) in enumerate(METRICS):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            for pipe, label, color, marker in SERIES:
                rs = sorted((r for r in rows if r["factor"] == f and r["pipeline"] == pipe), key=lambda r: r["level"])
                if metric == "mean_extra_time_s" and pipe == "ignore":
                    continue  # ignoring debris changes the route only by being knocked about; not a cost to plot
                ax.fill_between(x, np.array([r[lo_k] for r in rs]) * scale, np.array([r[hi_k] for r in rs]) * scale,
                                color=color, alpha=0.14, lw=0)
                ax.plot(x, np.array([r[metric] for r in rs]) * scale, color=color, lw=2, marker=marker, ms=7,
                        mec=SURFACE, mew=1.5, label=label)
            if metric == "median_gap_m":
                ax.axhline(SAFETY, color=MUTED, lw=1, ls="--")
                ax.text(x[-1], SAFETY, "required gap", ha="right", va="bottom", fontsize=8, color=MUTED)
            if scale == 100:
                ax.set_ylim(-3, 103)
            ax.set_xticks(x, [f"{v:g}" for v in levels])
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
    # Direct labels on the top-left panel, where the two lines are far apart.
    for pipe, label, *_ in SERIES:
        r = next(r for r in rows if r["factor"] == "speed" and r["pipeline"] == pipe and r["level"] == 0.1)
        # Inside the panel: under the line pinned at 100%, over the line pinned at 0%.
        axes[0, 0].annotate(label, (1, r["collision_rate"] * 100), xytext=(0, -16 if pipe == "ignore" else 9),
                            textcoords="offset points", ha="center", fontsize=9, color=INK)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False, fontsize=10,
               labelcolor=INK)
    fig.suptitle("Debris aimed at the robot", x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.005, f"{runs} encounters per level; each flown both ways on the same spacecraft, debris "
             "aimed from a clean run of the same mission. Bands: 95% CI (rates, time) or 10th–90th percentile (gap).",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def table(rows, results):
    lines = ["# Hazard sweep", ""]
    for f, (title, xlabel, levels) in FACTORS.items():
        lines += [f"## {title}", "",
                  f"| {xlabel} | Hit: ignore → avoid | Median gap (avoid) | Kept 0.5 m gap | Warning (median s) "
                  f"| Extra time | Extra fuel (pts) | No safe option | Success: ignore → avoid |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for v in levels:
            ig = next(r for r in rows if r["factor"] == f and r["level"] == v and r["pipeline"] == "ignore")
            av = next(r for r in rows if r["factor"] == f and r["level"] == v and r["pipeline"] == "avoid")
            lines.append(f"| {v:g} | {100 * ig['collision_rate']:.0f}% → {100 * av['collision_rate']:.0f}% | "
                         f"{av['median_gap_m']:.2f} m | {100 * av['kept_safety_rate']:.0f}% | {av['median_warning_s']:.0f} | "
                         f"+{av['mean_extra_time_s']:.0f} s | +{av['mean_extra_propellant_pct']:.1f} | "
                         f"{100 * av['no_safe_option_rate']:.0f}% | {100 * ig['success_rate']:.0f}% → {100 * av['success_rate']:.0f}% |")
        lines.append("")
    moving = [r for r in results if r["robot_moving"]]
    lines.append(f"Debris aimed at a robot in transit in {len(moving)} of {len(results)} encounters "
                 f"(the rest: while holding at a viewpoint).")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
