"""Baseline 1: fixed-route inspection.

Visit every target in a fixed order, hold at each viewpoint, return home.
This is the yardstick every smarter planner will be compared against.
"""

import numpy as np

from ..controllers.pose_controller import PoseController
from ..core.types import VehicleSpec
from .paths import plan_path
from .sequencer import Leg, LegSequencer


def plan_fixed_route(targets, obstacles, home, *, order=None, margin=0.6, keepout_margin=0.8,
                     dwell=4.0, center=np.zeros(3)) -> list[Leg]:
    """
    targets   inspection targets (from the spacecraft's known model)
    obstacles boxes to stay `margin` metres away from (measured to the robot's centre)
    home      start/end position
    order     target names in visiting order (default: as listed)
    """
    by_name = {t.name: t for t in targets}
    ordered = [by_name[n] for n in order] if order else list(targets)
    safe_radius = max(np.linalg.norm(b.center - center) + b.bounding_radius for b in obstacles) + keepout_margin

    legs, here = [], np.asarray(home, float)
    for t in ordered:
        path = plan_path(here, t.viewpoint, obstacles, safe_radius, margin, center)
        legs.append(Leg(t.name, path, look_at=t.point, transit_look_at=center, dwell=dwell, kind="inspect"))
        here = t.viewpoint
    path = plan_path(here, home, obstacles, safe_radius, margin, center)
    legs.append(Leg("home", path, look_at=center, transit_look_at=center, kind="return"))
    return legs


def fixed_route_inspector(vehicle: VehicleSpec, targets, obstacles, home, **kwargs) -> LegSequencer:
    margin = vehicle.bounding_radius + 0.35  # robot's own size + a safety buffer
    legs = plan_fixed_route(targets, obstacles, home, margin=margin, **kwargs)
    return LegSequencer(legs, PoseController(vehicle))
