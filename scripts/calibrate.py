"""Fit the confidence model on one set of missions, measure its calibration on another.

Needs per-inspection data from the noise sweep (registered pipeline):
    python scripts/noise_sweep.py --pipelines registered --runs 40 --first-seed 3000 --out results/calib_train
    python scripts/noise_sweep.py --pipelines registered --runs 20 --first-seed 4000 --out results/calib_test
    python scripts/calibrate.py

Writes the model to space_autonomy/perception/confidence_model.json (the agent
loads it from there) and a report + plot to results/calibration/.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from space_autonomy.perception.calibration import (DEFAULT_MODEL, FEATURES, ConfidenceModel, auroc, brier,
                                                   expected_calibration_error, reliability)

SERIES = [("calibrated", "Learned (calibrated)", "#2a78d6", "o"),
          ("heuristic", "Heuristic (before)", "#eb6834", "^")]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e0", "#fcfcfb"
TEXT = {"verdict", "truth", "component", "mission"}


def load(path):
    with open(path) as fh:
        return [{k: v if k in TEXT else float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def condition(r):
    if r["nav_pos_sigma"]:
        return ("nav_pos", r["nav_pos_sigma"] * 100)
    if r["nav_att_sigma_deg"]:
        return ("nav_att", r["nav_att_sigma_deg"])
    if r["sun_error_deg"]:
        return ("sun", r["sun_error_deg"])
    return ("ideal", 0.0)


def metrics(conf, correct, threshold=0.9):
    conf, correct = np.asarray(conf), np.asarray(correct)
    trusted = conf >= threshold
    return {"ece": expected_calibration_error(conf, correct), "brier": brier(conf, correct),
            "auroc": auroc(conf, correct), "accuracy": float(correct.mean()), "mean_conf": float(conf.mean()),
            "trusted_share": float(trusted.mean()),
            "trusted_accuracy": float(correct[trusted].mean()) if trusted.any() else float("nan"),
            "untrusted_accuracy": float(correct[~trusted].mean()) if (~trusted).any() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="results/calib_train/assessments_registered.csv")
    ap.add_argument("--test", default="results/calib_test/assessments_registered.csv")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, test = load(args.train), load(args.test)
    model = ConfidenceModel.fit(train, l2=args.l2)
    model.save(args.model)
    for r in test:
        r["calibrated"] = model.predict(r, r["verdict"])
        r["heuristic"] = r["heuristic_confidence"]

    correct = [r["correct"] for r in test]
    m = {k: metrics([r[k] for r in test], correct) for k, *_ in SERIES}

    by_cond = {}
    for r in test:
        by_cond.setdefault(condition(r), []).append(r)
    cond_rows = []
    for (f, v), rs in sorted(by_cond.items()):
        c = [r["correct"] for r in rs]
        cond_rows.append({"factor": f, "level": v, "n": len(rs), "accuracy": float(np.mean(c)),
                          **{f"mean_{k}": float(np.mean([r[k] for r in rs])) for k, *_ in SERIES}})

    # Distribution shift: train without one kind of noise at all, test only on that kind.
    shift = []
    for factor in ("nav_pos", "nav_att", "sun"):
        held = ConfidenceModel.fit([r for r in train if condition(r)[0] != factor], l2=args.l2)
        rs = [r for r in test if condition(r)[0] == factor]
        conf = [held.predict(r, r["verdict"]) for r in rs]
        c = [r["correct"] for r in rs]
        shift.append({"factor": factor, "n": len(rs), "accuracy": float(np.mean(c)), "mean_conf": float(np.mean(conf)),
                      "ece_heuristic": expected_calibration_error([r["heuristic"] for r in rs], c),
                      "ece": expected_calibration_error(conf, c), "auroc": auroc(conf, c)})

    plot(test, cond_rows, out / "calibration.png", len(train))
    report = make_report(m, cond_rows, model, len(train), len(test), args, shift)
    (out / "summary.md").write_text(report + "\n")
    print(report)
    print(f"\nModel saved to {args.model}; report and plot in {out}/")


def plot(test, cond_rows, path, n_train):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor(SURFACE)
    correct = [r["correct"] for r in test]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    ax = axes[0]
    for key, label, color, marker in SERIES:
        rel = reliability([r[key] for r in test], correct)
        x, y, n = (np.array(v) for v in zip(*rel))
        ax.plot(x, y, color=color, lw=2, zorder=2)
        ax.scatter(x, y, s=20 + 180 * np.sqrt(n / n.max()), color=color, marker=marker, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=label)
    ax.set_title("Reliability: stated confidence vs. how often it was right", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("stated confidence (bin average)", color=MUTED)
    ax.set_ylabel("fraction of verdicts that were correct", color=MUTED)
    ax.text(0.98, 0.04, "marker size = number of verdicts", ha="right", color=MUTED, fontsize=8,
            transform=ax.transAxes)

    ax = axes[1]
    for key, label, color, marker in SERIES:
        ax.scatter([c[f"mean_{key}"] for c in cond_rows], [c["accuracy"] for c in cond_rows], s=55,
                   color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
    worst = min(cond_rows, key=lambda c: c["accuracy"])
    ax.annotate(f"{worst['factor'].replace('_', ' ')} {worst['level']:g}", (worst["mean_heuristic"], worst["accuracy"]),
                xytext=(-8, 6), textcoords="offset points", ha="right", fontsize=8, color=INK)
    ax.annotate("", (worst["mean_calibrated"], worst["accuracy"]), (worst["mean_heuristic"], worst["accuracy"]),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_title("Per condition: average confidence vs. accuracy", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("average stated confidence", color=MUTED)
    ax.set_ylabel("accuracy under that condition", color=MUTED)
    ax.text(0.98, 0.04, f"one point per noise condition\n({len(cond_rows)} conditions, held-out missions)", ha="right",
            color=MUTED, fontsize=8, transform=ax.transAxes)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False, fontsize=10,
               labelcolor=INK, markerscale=0.7)
    fig.suptitle("Is the robot's confidence honest?", x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.01, f"Dashed line = perfectly calibrated. Fitted on {n_train} inspections (seeds 3000+), "
             "evaluated on different missions (seeds 4000+).", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_report(m, cond_rows, model, n_train, n_test, args, shift):
    lines = [
        "# Confidence calibration", "",
        f"Fitted on {n_train} inspections ({args.train}), evaluated on {n_test} held-out inspections ({args.test}).", "",
        "| Metric | Heuristic (before) | Learned (calibrated) | Better is |",
        "|---|---|---|---|",
        f"| Expected calibration error | {m['heuristic']['ece']:.3f} | {m['calibrated']['ece']:.3f} | lower |",
        f"| Brier score | {m['heuristic']['brier']:.3f} | {m['calibrated']['brier']:.3f} | lower |",
        f"| AUROC (right vs. wrong) | {m['heuristic']['auroc']:.3f} | {m['calibrated']['auroc']:.3f} | higher |",
        f"| Mean confidence / actual accuracy | {m['heuristic']['mean_conf']:.3f} / {m['heuristic']['accuracy']:.3f} "
        f"| {m['calibrated']['mean_conf']:.3f} / {m['calibrated']['accuracy']:.3f} | equal |",
        f"| Verdicts at >= 90% confidence | {100 * m['heuristic']['trusted_share']:.0f}%, "
        f"{100 * m['heuristic']['trusted_accuracy']:.1f}% right | {100 * m['calibrated']['trusted_share']:.0f}%, "
        f"{100 * m['calibrated']['trusted_accuracy']:.1f}% right | |",
        f"| Verdicts below 90% | {100 * m['heuristic']['untrusted_accuracy']:.1f}% right "
        f"| {100 * m['calibrated']['untrusted_accuracy']:.1f}% right | |",
        "", "## Per condition (held-out)", "",
        "| Condition | Inspections | Accuracy | Mean heuristic confidence | Mean calibrated confidence |",
        "|---|---|---|---|---|",
    ]
    for c in cond_rows:
        lines.append(f"| {c['factor']} {c['level']:g} | {c['n']} | {c['accuracy']:.2f} | {c['mean_heuristic']:.2f} "
                     f"| {c['mean_calibrated']:.2f} |")
    lines += ["", "## Distribution shift: leave one kind of noise out of training", "",
              "Trained with no examples of that noise at all, tested only on it.", "",
              "| Held-out noise | Inspections | Accuracy | Mean confidence | ECE heuristic | ECE learned | AUROC learned |",
              "|---|---|---|---|---|---|---|"]
    for r in shift:
        lines.append(f"| {r['factor']} | {r['n']} | {r['accuracy']:.2f} | {r['mean_conf']:.2f} | "
                     f"{r['ece_heuristic']:.3f} | {r['ece']:.3f} | {r['auroc']:.3f} |")
    lines += ["", "## What each per-verdict model leans on", "",
              "Standardised weights (positive = more likely correct). Largest three by size:", ""]
    for verdict, mdl in model.models.items():
        if "constant" in mdl:
            lines.append(f"- **{verdict}** ({mdl['rows']} training rows): constant {mdl['constant']:.3f}")
            continue
        top = sorted(zip(FEATURES, mdl["weights"]), key=lambda kv: -abs(kv[1]))[:3]
        lines.append(f"- **{verdict}** ({mdl['rows']} training rows): "
                     + ", ".join(f"{k} {w:+.2f}" for k, w in top))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
