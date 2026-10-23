"""Target spacecraft built from primitive shapes.

A spacecraft is a list of components. From that one list we generate both
the MuJoCo model (what the simulator renders and collides with) and the
geometry the autonomy plans around (obstacle boxes and inspection targets).
Later, randomised spacecraft (M10) are just different component lists.
"""

from dataclasses import dataclass

import numpy as np

from ..core.geometry import Box, InspectionTarget
from ..core.rotations import look_at_quat, quat_to_mat


@dataclass(frozen=True)
class Component:
    name: str
    shape: str                       # "box" or "cylinder"
    pos: tuple                       # m, spacecraft frame
    size: tuple                      # box: half-extents (x, y, z); cylinder: (radius, half-length)
    rgba: tuple
    axis: tuple = (0.0, 0.0, 1.0)    # cylinder axis direction
    inspect_from: tuple | None = None  # direction to view it from; None = not an inspection target
    standoff: float = 1.5            # m

    @property
    def rotation(self) -> np.ndarray:
        # A frame whose z axis is `axis` (any roll is fine for a cylinder).
        R = quat_to_mat(look_at_quat(self.axis))  # body x -> axis
        return np.column_stack([R[:, 1], R[:, 2], R[:, 0]])

    def obstacle(self) -> Box:
        """Conservative box around the component (a cylinder becomes its bounding box)."""
        if self.shape == "box":
            half = np.array(self.size, float)
        else:
            r, h = self.size
            half = np.array([r, r, h], float)
        return Box(np.array(self.pos, float), half, self.rotation)

    def target(self) -> InspectionTarget | None:
        if self.inspect_from is None:
            return None
        n = np.array(self.inspect_from, float)
        return InspectionTarget(self.name, np.array(self.pos, float), n / np.linalg.norm(n), self.standoff)

    def mjcf(self) -> str:
        size = " ".join(f"{s:g}" for s in self.size)
        pos = " ".join(f"{p:g}" for p in self.pos)
        rgba = " ".join(f"{c:g}" for c in self.rgba)
        orient = "" if self.shape == "box" else f' zaxis="{" ".join(f"{a:g}" for a in self.axis)}"'
        return (f'<geom name="sc_{self.name}" type="{self.shape}" size="{size}" pos="{pos}"{orient} '
                f'rgba="{rgba}"/>')


@dataclass(frozen=True)
class Spacecraft:
    name: str
    components: tuple

    def obstacles(self) -> list[Box]:
        return [c.obstacle() for c in self.components]

    def targets(self) -> list[InspectionTarget]:
        return [t for c in self.components if (t := c.target()) is not None]

    def bounding_radius(self) -> float:
        return max(np.linalg.norm(b.center) + b.bounding_radius for b in self.obstacles())

    def mjcf(self) -> str:
        geoms = "\n      ".join(c.mjcf() for c in self.components)
        # No joint: the spacecraft is welded to the world. We simulate the
        # *relative* environment, so the target is the fixed reference frame.
        return f'<body name="spacecraft" pos="0 0 0">\n      {geoms}\n    </body>'
