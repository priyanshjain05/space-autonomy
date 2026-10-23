"""What the robot believes about the spacecraft, component by component.

This is the robot's memory of the mission. Today it only records the most
recent observation of each component. It is also what a smarter planner
(M7) will read to decide where to look next: anything still
"not_inspected" or low-confidence is worth a visit.
"""

from dataclasses import asdict, dataclass

from ..perception.change_detection import NOMINAL, UNOBSERVABLE, Assessment

NOT_INSPECTED, OK, ANOMALY = "not_inspected", "nominal", "anomaly"


@dataclass
class Belief:
    component: str
    status: str = NOT_INSPECTED   # not_inspected | nominal | anomaly | unobservable
    kind: str | None = None       # anomaly type, when status == anomaly
    confidence: float = 0.0       # how sure we are about `status`
    observations: int = 0


class WorldModel:
    def __init__(self, components):
        self.beliefs = {c: Belief(c) for c in components}

    def update(self, a: Assessment):
        b = self.beliefs[a.component]
        b.observations += 1
        b.confidence = round(a.confidence, 3)
        if a.kind == NOMINAL:
            b.status, b.kind = OK, None
        elif a.kind == UNOBSERVABLE:
            b.status, b.kind = UNOBSERVABLE, None
        else:
            b.status, b.kind = ANOMALY, a.kind

    def __getitem__(self, name) -> Belief:
        return self.beliefs[name]

    def anomalies(self) -> list[Belief]:
        return [b for b in self.beliefs.values() if b.status == ANOMALY]

    def table(self, title="") -> str:
        lines = [title, f"{'Component':<16}{'State':<16}{'Finding':<17}{'Confidence':>10}", "─" * 59]
        for b in self.beliefs.values():
            finding = b.kind or "-"
            conf = f"{b.confidence:.2f}" if b.observations else "-"
            lines.append(f"{b.component:<16}{b.status:<16}{finding:<17}{conf:>10}")
        return "\n".join(l for l in lines if l)

    def to_dict(self) -> dict:
        return {k: asdict(v) for k, v in self.beliefs.items()}
