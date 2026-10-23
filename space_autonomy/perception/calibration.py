"""Calibrated confidence: how likely is this verdict to be right?

"Calibrated" means: of all the verdicts the robot reports at 80% confidence,
about 80% are correct. The heuristic in change_detection.py is not: it says
100% whenever there's a lot of changed pixels, including when those pixels
are just misregistration strips.

The simulator gives labelled data for free: every inspection comes with the
true answer. So we learn the confidence instead of guessing it: one small
logistic regression per verdict ("missing", "nominal", ...), mapping things
the robot can observe to P(verdict is correct):

    evidence size     how many pixels changed, relative to the detection threshold
    evidence shape    share of change on predicted edges, thickness of the thickest blob
    registration      did the 3D/2D alignment succeed, how well did it fit, how big a correction

Logistic regression is deliberately simple: a handful of interpretable
weights, trains in milliseconds, and its outputs are probabilities by design.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).with_name("confidence_model.json")

FEATURES = ("far", "near", "color", "expected", "edge_frac", "thickness",
            "registered", "icp_accepted", "icp_fit_cm", "icp_gain", "image_accepted",
            "image_shift_px", "image_rot_deg", "fallback")


def features(a, min_px=250, reg=None, img=None, fallback=False) -> dict:
    """Observable facts about one inspection, as numbers (no ground truth in here)."""
    f = {
        "far": np.log1p(a.farther_px / min_px),
        "near": np.log1p(a.nearer_px / min_px),
        "color": np.log1p(a.color_px / min_px),
        "expected": np.log1p(a.expected_px / 1000),
        "edge_frac": a.edge_frac,
        "thickness": np.log1p(a.thickness_px),
        "registered": 0.0, "icp_accepted": 0.0, "icp_fit_cm": 0.0, "icp_gain": 0.0,
        "image_accepted": 0.0, "image_shift_px": 0.0, "image_rot_deg": 0.0, "fallback": 0.0,
    }
    if reg is not None:
        before, after = reg.rms_before_m, reg.rms_after_m
        f.update(registered=1.0, icp_accepted=float(reg.accepted), fallback=float(fallback),
                 icp_fit_cm=100 * after if np.isfinite(after) else 15.0,
                 icp_gain=(before - after) / before if np.isfinite(before) and before > 0 else 0.0,
                 image_accepted=float(img.accepted), image_shift_px=float(np.hypot(img.dx_px, img.dy_px)),
                 image_rot_deg=abs(img.rotation_deg))
    return {k: float(v) for k, v in f.items()}


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, l2=1.0, iters=50):
    """Logistic regression by Newton's method, features standardised, L2 penalty (not on the bias)."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-6] = 1.0  # a feature that never varied in training carries no information; don't divide by ~0
    Z = np.column_stack([np.ones(len(X)), (X - mean) / std])
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], l2)
    pen[0] = 0
    for _ in range(iters):
        p = _sigmoid(Z @ w)
        grad = Z.T @ (p - y) + pen * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mean": mean.tolist(), "std": std.tolist(), "bias": float(w[0]), "weights": w[1:].tolist()}


@dataclass
class ConfidenceModel:
    models: dict          # verdict -> fitted logistic regression, or {"constant": p}
    min_rows: int = 40

    def predict(self, f: dict, verdict: str) -> float:
        m = self.models.get(verdict)
        if m is None:
            return 0.5  # never seen this verdict in training: no idea
        if "constant" in m:
            return m["constant"]
        x = (np.array([f[k] for k in FEATURES]) - m["mean"]) / m["std"]
        return float(_sigmoid(m["bias"] + x @ np.array(m["weights"])))

    @classmethod
    def fit(cls, rows, l2=1.0, min_rows=40) -> "ConfidenceModel":
        """rows: dicts with the FEATURES plus "verdict" and "correct" (0/1)."""
        models = {}
        for verdict in sorted({r["verdict"] for r in rows}):
            rs = [r for r in rows if r["verdict"] == verdict]
            y = np.array([float(r["correct"]) for r in rs])
            if len(rs) < min_rows or y.min() == y.max():
                # Too little data, or never wrong / never right: use the smoothed rate.
                models[verdict] = {"constant": float((y.sum() + 1) / (len(y) + 2)), "rows": len(rs)}
                continue
            X = np.array([[r[k] for k in FEATURES] for r in rs], float)
            models[verdict] = {**fit_logistic(X, y, l2), "rows": len(rs)}
        return cls(models, min_rows)

    def save(self, path=DEFAULT_MODEL):
        Path(path).write_text(json.dumps({"features": FEATURES, "models": self.models}, indent=1) + "\n")

    @classmethod
    def load(cls, path=DEFAULT_MODEL) -> "ConfidenceModel | None":
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if tuple(data["features"]) != FEATURES:
            raise ValueError(f"{path} was trained on different features; re-run scripts/calibrate.py")
        return cls(data["models"])


# ── Measuring calibration ──────────────────────────────────────────────────


def reliability(conf, correct, bins=10):
    """Per confidence bin: (mean confidence, accuracy, count). Calibrated = the first two match."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return [(conf[idx == b].mean(), correct[idx == b].mean(), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]


def expected_calibration_error(conf, correct, bins=10) -> float:
    """Average gap between stated confidence and actual accuracy, weighted by how many verdicts fall in each bin."""
    n = len(conf)
    return float(sum(c * abs(m - acc) for m, acc, c in reliability(conf, correct, bins)) / n)


def brier(conf, correct) -> float:
    """Mean squared error of the confidence as a probability (lower is better; 0.25 = coin flip at 50%)."""
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def auroc(conf, correct) -> float:
    """Chance a random correct verdict gets higher confidence than a random wrong one (0.5 = useless)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks for ties
        ranks[allv == v] = ranks[allv == v].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
