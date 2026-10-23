"""Shapes the autonomy reasons about: obstacle boxes and inspection targets."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Oriented box. `rotation` maps box-local axes into the world frame."""
    center: np.ndarray
    half: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def distance(self, p) -> float:
        """Signed distance from point p to the box surface (negative inside)."""
        local = self.rotation.T @ (np.asarray(p, float) - self.center)
        q = np.abs(local) - self.half
        return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0))

    @property
    def bounding_radius(self) -> float:
        return float(np.linalg.norm(self.half))


def clearance(p, boxes) -> float:
    return min(b.distance(p) for b in boxes)


@dataclass(frozen=True)
class InspectionTarget:
    """A component to look at, and the direction to look at it from."""
    name: str
    point: np.ndarray     # what to look at
    normal: np.ndarray    # unit vector pointing from the component toward the viewer
    standoff: float       # m, how far away to hold

    @property
    def viewpoint(self) -> np.ndarray:
        return self.point + self.standoff * self.normal
