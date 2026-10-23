"""Collision-free paths around the spacecraft.

Rule of thumb borrowed from real proximity operations: there is a
"keep-out sphere" around the target that you only enter on purpose,
along a known corridor. So:

1. If the straight line between two points keeps a safe distance from
   every part of the spacecraft, fly it.
2. Otherwise, back straight out to the keep-out sphere, travel around
   the sphere's surface (a great-circle arc), and come straight back in.

It is not optimal (it wastes fuel on detours), but it is predictable and
easy to verify, which is what a baseline should be.
"""

import numpy as np

from ..core.geometry import clearance


class PlanningError(RuntimeError):
    pass


def segment_clear(a, b, obstacles, margin, step=0.05) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    return all(clearance(a + (b - a) * s, obstacles) > margin for s in np.linspace(0, 1, n))


def _slerp_dirs(u, v, n):
    """n+1 unit vectors evenly spaced along the great circle from u to v."""
    angle = np.arccos(np.clip(u @ v, -1, 1))
    if angle < 1e-6:
        return [u] * (n + 1)
    if np.pi - angle < 1e-3:  # opposite points: any great circle works, pick one through +z
        side = np.cross(u, [0, 0, 1.0])
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(u, [1.0, 0, 0])
        w = np.cross(side / np.linalg.norm(side), u)
    else:
        w = v - (u @ v) * u
        w /= np.linalg.norm(w)
    return [np.cos(angle * k / n) * u + np.sin(angle * k / n) * w for k in range(n + 1)]


def plan_path(a, b, obstacles, safe_radius, margin, center=np.zeros(3), spacing=0.6) -> list:
    """Waypoints from a to b (a excluded, b included)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if segment_clear(a, b, obstacles, margin):
        return [b]

    ra, rb = a - center, b - center
    ua, ub = ra / np.linalg.norm(ra), rb / np.linalg.norm(rb)
    # Points outside the sphere drop straight in to it; outside is clear by construction.
    radius = safe_radius
    angle = np.arccos(np.clip(ua @ ub, -1, 1))
    n = max(1, int(np.ceil(radius * angle / spacing)))
    arc = [center + radius * d for d in _slerp_dirs(ua, ub, n)]

    path = [p for p in arc + [b] if np.linalg.norm(p - a) > 1e-6]
    for p, q in zip([a] + path[:-1], path):
        if not segment_clear(p, q, obstacles, margin):
            raise PlanningError(f"no safe corridor from {np.round(a, 2)} to {np.round(b, 2)}")
    return path
