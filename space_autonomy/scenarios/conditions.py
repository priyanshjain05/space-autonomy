"""Operating conditions: how far reality departs from what the robot assumes."""

from dataclasses import dataclass

import numpy as np

from ..core.rotations import quat_from_axis_angle, quat_to_mat
from ..sim.sensors import NavigationError
from ..sim.world import SUN_DIR


@dataclass(frozen=True)
class Conditions:
    nav_pos_sigma: float = 0.0      # m per axis, navigation position error
    nav_att_sigma_deg: float = 0.0  # deg per axis, navigation attitude error
    sun_error_deg: float = 0.0      # angle between the real sun and the sun the digital twin assumes

    def navigation(self, seed) -> NavigationError | None:
        if self.nav_pos_sigma == 0 and self.nav_att_sigma_deg == 0:
            return None
        return NavigationError(self.nav_pos_sigma, np.radians(self.nav_att_sigma_deg), seed=seed)

    def sun_dir(self, seed) -> np.ndarray:
        """The real sun: the nominal direction turned by sun_error_deg about a random perpendicular axis."""
        if self.sun_error_deg == 0:
            return SUN_DIR
        rng = np.random.default_rng(seed + 7919)
        axis = np.cross(SUN_DIR, rng.normal(size=3))
        return quat_to_mat(quat_from_axis_angle(axis, np.radians(self.sun_error_deg))) @ SUN_DIR

    def label(self) -> str:
        parts = [f"nav {100 * self.nav_pos_sigma:g} cm" if self.nav_pos_sigma else "",
                 f"att {self.nav_att_sigma_deg:g} deg" if self.nav_att_sigma_deg else "",
                 f"sun {self.sun_error_deg:g} deg" if self.sun_error_deg else ""]
        return ", ".join(p for p in parts if p) or "ideal"
