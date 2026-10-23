import numpy as np

from space_autonomy.perception.calibration import (FEATURES, ConfidenceModel, auroc, brier,
                                                   expected_calibration_error, fit_logistic)

rng = np.random.default_rng(0)


def synthetic_rows(n, verdict="nominal"):
    """Rows where P(correct) is exactly sigmoid(1 - 2 * edge_frac + thickness): a known answer to recover."""
    rows = []
    for _ in range(n):
        f = {k: float(rng.normal()) for k in FEATURES}
        p = 1 / (1 + np.exp(-(1.0 - 2.0 * f["edge_frac"] + 1.0 * f["thickness"])))
        rows.append({**f, "verdict": verdict, "correct": int(rng.random() < p), "p_true": p})
    return rows


def test_logistic_regression_recovers_known_weights():
    rows = synthetic_rows(20000)
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    y = np.array([r["correct"] for r in rows], float)
    m = fit_logistic(X, y, l2=0.1)
    w = dict(zip(FEATURES, np.array(m["weights"]) / np.array(m["std"])))  # back to raw units
    assert abs(w["edge_frac"] + 2.0) < 0.15
    assert abs(w["thickness"] - 1.0) < 0.15
    assert abs(w["far"]) < 0.1


def test_fitted_model_is_calibrated_on_fresh_data():
    model = ConfidenceModel.fit(synthetic_rows(8000))
    test = synthetic_rows(8000)
    conf = [model.predict(r, "nominal") for r in test]
    correct = [r["correct"] for r in test]
    assert expected_calibration_error(conf, correct) < 0.03
    assert abs(np.mean(conf) - np.mean(correct)) < 0.02


def test_metrics_on_known_cases():
    correct = np.array([1, 1, 0, 0])
    assert expected_calibration_error([1, 1, 0, 0], correct) == 0
    assert expected_calibration_error([1, 1, 1, 1], correct) == 0.5  # always 100% sure, right half the time
    assert brier([0.5] * 4, correct) == 0.25
    assert auroc([0.9, 0.8, 0.2, 0.1], correct) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], correct) == 0.0
    assert auroc([0.5] * 4, correct) == 0.5


def test_rare_or_one_sided_verdicts_fall_back_to_a_smoothed_rate():
    rows = [{**{k: 0.0 for k in FEATURES}, "verdict": "missing", "correct": 1} for _ in range(100)]
    rows += [{**{k: 0.0 for k in FEATURES}, "verdict": "surface_damage", "correct": i % 2} for i in range(10)]
    model = ConfidenceModel.fit(rows)
    assert np.isclose(model.predict(rows[0], "missing"), 101 / 102)  # never wrong in training, but not 100%
    assert np.isclose(model.predict(rows[0], "surface_damage"), 6 / 12)
    assert model.predict(rows[0], "never_seen") == 0.5


def test_save_load_round_trip(tmp_path):
    model = ConfidenceModel.fit(synthetic_rows(500))
    path = tmp_path / "m.json"
    model.save(path)
    again = ConfidenceModel.load(path)
    r = synthetic_rows(1)[0]
    assert np.isclose(model.predict(r, "nominal"), again.predict(r, "nominal"))


def test_shipped_model_doubts_misregistration_but_trusts_real_faults():
    """The trained model (perception/confidence_model.json), through the agent's real pipeline.

    A healthy docking port seen with 2.5 deg of attitude error: too much for registration, so the
    detector wrongly reports "misaligned" -- and the old heuristic is 100% sure. Then real debris.
    """
    from dataclasses import replace

    from space_autonomy.autonomy.agent import InspectionAgent
    from space_autonomy.autonomy.world_model import WorldModel
    from space_autonomy.core.rotations import look_at_quat, quat_from_axis_angle, quat_mul
    from space_autonomy.core.types import VehicleSpec
    from space_autonomy.perception.reference import ReferenceRenderer
    from space_autonomy.scenarios.alpha03 import ALPHA03
    from space_autonomy.sim.anomalies import inject
    from space_autonomy.sim.sensors import OnboardCamera
    from space_autonomy.sim.world import SpaceSim

    model = ConfidenceModel.load()
    assert model is not None, "run scripts/calibrate.py to train the model"
    vehicle = VehicleSpec()
    reference = ReferenceRenderer(vehicle, ALPHA03)
    targets = {t.name: t for t in ALPHA03.targets()}

    def inspect(spacecraft, name, att_deg):
        t = targets[name]
        sim = SpaceSim(vehicle, spacecraft, start_pos=t.viewpoint, start_quat=look_at_quat(t.point - t.viewpoint))
        obs, true = OnboardCamera(sim, seed=0).capture(), sim.state()
        est = replace(true, quat=quat_mul(true.quat, quat_from_axis_angle([0.3, -0.5, 0.8], np.radians(att_deg))))
        agent = InspectionAgent(None, reference, WorldModel([name]), confidence_model=model)
        agent.image_request = name
        agent.on_image(obs, est)
        return agent.pop_events()[0]

    e = inspect(ALPHA03, "docking_port", 2.5)
    assert e["result"] == "misaligned" and e["heuristic_confidence"] > 0.95  # wrong, and sure of it
    assert e["confidence"] < 0.2

    debris, _ = inject(ALPHA03, "foreign_object", "antenna", np.random.default_rng(0))
    e = inspect(debris, "antenna", 0.3)
    assert e["result"] == "foreign_object" and e["confidence"] > 0.9
