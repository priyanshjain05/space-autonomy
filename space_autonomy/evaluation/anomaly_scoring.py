"""Compare what the robot believes with what was really injected."""

from ..autonomy.world_model import ANOMALY, NOT_INSPECTED


def score_anomalies(world_model, truth, targets) -> dict:
    """
    found           injected anomalies the robot flagged (any type)
    type_correct    ... and named the right type
    false_alarms    healthy components the robot flagged
    """
    truth_by = {a.component: a for a in truth}
    details, found, correct = [], 0, 0
    for name, a in truth_by.items():
        b = world_model[name]
        hit = b.status == ANOMALY
        found += hit
        correct += hit and b.kind == a.kind
        details.append({"component": name, "injected": a.kind, "detail": a.detail,
                        "reported": b.kind if hit else b.status, "confidence": b.confidence})
    false_alarms = [n for n in targets if n not in truth_by and world_model[n].status == ANOMALY]
    return {
        "anomalies_injected": len(truth_by),
        "anomalies_found": found,
        "anomaly_type_correct": correct,
        "false_alarms": len(false_alarms),
        "components_not_inspected": sum(world_model[n].status == NOT_INSPECTED for n in targets),
        "anomaly_details": details,
    }
