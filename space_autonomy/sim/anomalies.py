"""Anomaly injection: turn the nominal spacecraft into the one the robot actually meets.

The robot is told what the spacecraft *should* look like (the nominal
model). The simulator builds a modified copy with some of these faults:

    missing         a component is gone (e.g. antenna broke off)
    misaligned      a component is rotated 20-30 deg from where it belongs
                    (e.g. solar array drive stuck at the wrong angle)
    foreign_object  a piece of debris is stuck to a component
    surface_damage  a scorch mark on a component's surface

The ground truth is kept so the evaluation can check what the robot found.
"""

from dataclasses import dataclass, replace

import numpy as np

from ..core.rotations import mat_to_quat, quat_from_axis_angle, quat_mul, quat_to_mat
from .spacecraft import Component, Spacecraft

KINDS = ("missing", "misaligned", "foreign_object", "surface_damage")

DEBRIS = (0.78, 0.78, 0.82, 1)    # torn insulation / metal fragment
SCORCH = (0.16, 0.11, 0.08, 1)


@dataclass(frozen=True)
class Anomaly:
    kind: str
    component: str
    detail: str


def _face(c: Component, n):
    """The flat face of c that points most toward direction n.

    Returns (centre, outward normal, the two in-plane axes, their half-extents).
    """
    R, half = c.rotation, c.obstacle().half
    local = R.T @ n
    k = int(np.argmax(np.abs(local)))
    normal = np.sign(local[k]) * R[:, k]
    others = [i for i in range(3) if i != k]
    return np.array(c.pos) + normal * half[k], normal, R[:, others], half[others]


def _support(R, half, n):
    """How far a box (rotation R, half-extents) reaches along direction n from its centre."""
    return float(np.abs(R.T @ n) @ half)


def _random_perpendicular(n, rng):
    v = np.cross(n, rng.normal(size=3))
    return v / np.linalg.norm(v)


def eligible_kinds(c: Component) -> list[str]:
    kinds = ["missing", "misaligned", "foreign_object"]
    # A dark scorch mark is invisible on an already-dark part, so only light parts can show damage.
    if np.dot(c.rgba[:3], (0.299, 0.587, 0.114)) > 0.3:
        kinds.append("surface_damage")
    return kinds


def inject(sc: Spacecraft, kind: str, name: str, rng) -> tuple[Spacecraft, Anomaly]:
    comps = list(sc.components)
    i = next(i for i, c in enumerate(comps) if c.name == name)
    c = comps[i]
    view = c.target().normal

    if kind == "missing":
        del comps[i]
        return replace(sc, components=tuple(comps)), Anomaly(kind, name, "component absent")

    if kind == "misaligned":
        axis = _random_perpendicular(view, rng)  # tilt toward/away from the viewer, so it's observable
        deg = rng.uniform(20, 30) * rng.choice([-1, 1])
        comps[i] = replace(c, quat=tuple(quat_mul(quat_from_axis_angle(axis, np.radians(deg)), c.quat)))
        return replace(sc, components=tuple(comps)), Anomaly(kind, name, f"rotated {deg:+.0f} deg")

    centre, normal, axes, ext = _face(c, view)
    if kind == "foreign_object":
        half = rng.uniform(0.06, 0.1, size=3)
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        R = quat_to_mat(q)
        offset = axes @ (rng.uniform(-0.5, 0.5, size=2) * ext)
        pos = centre + offset + normal * _support(R, half, normal)  # resting on the surface
        comps.append(Component(f"debris_on_{name}", "box", tuple(pos), tuple(half), DEBRIS, quat=tuple(q)))
        return replace(sc, components=tuple(comps)), Anomaly(kind, name, f"{2 * half.max():.2f} m debris")

    if kind == "surface_damage":
        if c.shape == "cylinder":
            # Round face: keep the whole square patch inside the circle.
            r = ext[0]
            size = np.minimum(rng.uniform(0.08, 0.14, size=2), 0.4 * r)
            u = rng.normal(size=2)
            offset = axes @ (u / np.linalg.norm(u) * rng.uniform(0, 0.9 * r - np.linalg.norm(size)))
        else:
            size = np.minimum(rng.uniform(0.08, 0.14, size=2), 0.6 * ext)
            offset = axes @ (rng.uniform(-1, 1, size=2) * (ext - size) * 0.8)
        R = np.column_stack([axes[:, 0], np.cross(normal, axes[:, 0]), normal])
        pos = centre + offset + normal * 0.003
        comps.append(Component(f"damage_on_{name}", "box", tuple(pos), (size[0], size[1], 0.003), SCORCH,
                               quat=tuple(mat_to_quat(R))))
        return replace(sc, components=tuple(comps)), Anomaly(kind, name, f"{2 * size[0]:.2f} x {2 * size[1]:.2f} m scorch")

    raise ValueError(f"unknown anomaly kind {kind!r}")


def random_anomalies(sc: Spacecraft, rng, n: int) -> tuple[Spacecraft, tuple[Anomaly, ...]]:
    """Inject n anomalies on n different inspection targets."""
    targets = [c for c in sc.components if c.inspect_from is not None]
    chosen = rng.choice(len(targets), size=n, replace=False)
    truth = []
    for idx in chosen:
        c = targets[idx]
        sc, a = inject(sc, str(rng.choice(eligible_kinds(c))), c.name, rng)
        truth.append(a)
    return sc, tuple(truth)
