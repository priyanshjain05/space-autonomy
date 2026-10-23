"""Inspection routes and viewpoints.

Baseline 1 is the fixed route: visit every target in a fixed order, hold at
each viewpoint, return home. Active inspection (M7) adds detours on the fly:
when a look is inconclusive, go and look again from somewhere else. Both use
the same collision-free path planner.
"""

import numpy as np

from ..controllers.pose_controller import PoseController
from ..core.geometry import clearance
from ..core.types import VehicleSpec
from .paths import plan_path, segment_clear
from .sequencer import Leg, LegSequencer


class InspectionPlanner:
    def __init__(self, obstacles, margin=0.6, keepout_margin=0.8, center=np.zeros(3)):
        """
        obstacles  boxes to stay `margin` metres away from (measured to the robot's centre)
        """
        self.obstacles = obstacles
        self.margin = margin
        self.center = np.asarray(center, float)
        self.safe_radius = max(np.linalg.norm(b.center - self.center) + b.bounding_radius
                               for b in obstacles) + keepout_margin

    def path(self, a, b) -> list:
        return plan_path(a, b, self.obstacles, self.safe_radius, self.margin, self.center)

    def fixed_route(self, targets, home, order=None, dwell=4.0) -> list[Leg]:
        """Visit targets in `order` (names; default as listed), then return home."""
        by_name = {t.name: t for t in targets}
        ordered = [by_name[n] for n in order] if order else list(targets)
        legs, here = [], np.asarray(home, float)
        for t in ordered:
            legs.append(self.inspect_leg(t, t.viewpoint, here, dwell))
            here = t.viewpoint
        legs.append(self.leg_to(here, "home", home, look_at=self.center, kind="return"))
        return legs

    def inspect_leg(self, target, viewpoint, start, dwell=4.0, kind="inspect") -> Leg:
        return Leg(target.name, self.path(start, viewpoint), look_at=target.point, transit_look_at=self.center,
                   dwell=dwell, kind=kind)

    def leg_to(self, start, name, goal, look_at, kind="goto", dwell=0.0) -> Leg:
        return Leg(name, self.path(start, goal), look_at=look_at, transit_look_at=self.center, dwell=dwell, kind=kind)

    def alternate_viewpoints(self, target, tilt_deg=30.0, count=8, avoid=()) -> list:
        """Other places to look at `target` from: the usual direction tilted by `tilt_deg`,
        at the same distance, spread around a cone. Kept only if the robot fits there safely
        and nothing blocks the line of sight. In cone order; nearest_viewpoint picks the cheapest.
        """
        n = target.normal
        u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        tilt = np.radians(tilt_deg)
        own = [b for b in self.obstacles if b.distance(target.point) < 1e-6]  # the target's own box
        others = [b for b in self.obstacles if all(b is not o for o in own)]
        out = []
        for k in range(count):
            phi = 2 * np.pi * k / count
            d = np.cos(tilt) * n + np.sin(tilt) * (np.cos(phi) * u + np.sin(phi) * v)
            vp = target.point + target.standoff * d
            if any(np.linalg.norm(vp - a) < 0.2 for a in avoid):
                continue
            if clearance(vp, self.obstacles) <= self.margin + 0.3:
                continue
            # Line of sight: stop just short of the target, whose own box it must of course enter.
            if others and not segment_clear(vp, target.point + 0.15 * (vp - target.point), others, 0.02):
                continue
            out.append(vp)
        return out

    def nearest_viewpoint(self, target, here, **kwargs):
        """The reachable alternate viewpoint with the shortest flight from `here` (None if there's none)."""
        best, best_len = None, np.inf
        for vp in self.alternate_viewpoints(target, **kwargs):
            try:
                path = self.path(here, vp)
            except Exception:
                continue
            length = sum(np.linalg.norm(b - a) for a, b in zip([here] + path[:-1], path))
            if length < best_len:
                best, best_len = vp, length
        return best


def plan_fixed_route(targets, obstacles, home, *, order=None, margin=0.6, keepout_margin=0.8,
                     dwell=4.0, center=np.zeros(3)) -> list[Leg]:
    return InspectionPlanner(obstacles, margin, keepout_margin, center).fixed_route(targets, home, order, dwell)


def vehicle_margin(vehicle: VehicleSpec) -> float:
    return vehicle.bounding_radius + 0.35  # robot's own size + a safety buffer


def fixed_route_inspector(vehicle: VehicleSpec, targets, obstacles, home, **kwargs) -> LegSequencer:
    legs = plan_fixed_route(targets, obstacles, home, margin=vehicle_margin(vehicle), **kwargs)
    return LegSequencer(legs, PoseController(vehicle))
