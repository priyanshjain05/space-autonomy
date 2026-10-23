"""Digital twin: what the camera *should* see if the spacecraft matched its nominal model.

This is the one place the autonomy uses MuJoCo, and only as a 3D renderer
for the nominal model. It never reads the simulated world. Its inputs are the
nominal spacecraft spec (what the operator says the spacecraft looks like)
and the robot's own pose estimate. Any renderer that can draw the nominal CAD
model (pyrender, Blender, Isaac Sim) could stand in for it.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from ..core.types import RobotState, VehicleSpec
from ..sim.render import render_geom_ids, render_rgbd
from ..sim.spacecraft import Spacecraft
from ..sim.world import build_mjcf


@dataclass(frozen=True)
class ReferenceView:
    rgb: np.ndarray       # (H, W, 3) uint8
    depth: np.ndarray     # (H, W) float32, metres
    labels: np.ndarray    # (H, W) int, component index (see ReferenceRenderer.names), -1 = empty space


class ReferenceRenderer:
    def __init__(self, vehicle: VehicleSpec, nominal: Spacecraft):
        cam = vehicle.camera
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(vehicle, nominal, np.zeros(3), (1, 0, 0, 0), 0.01))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, cam.height, cam.width)
        self.names = [c.name for c in nominal.components]
        # MuJoCo geom id -> component index; anything else (the robot itself) -> -1.
        self._lut = np.full(self.model.ngeom + 1, -1)
        for i, name in enumerate(self.names):
            self._lut[self.model.geom(f"sc_{name}").id] = i

    def render(self, state: RobotState) -> ReferenceView:
        self.data.qpos[0:3] = state.position
        self.data.qpos[3:7] = state.quat
        mujoco.mj_forward(self.model, self.data)
        rgb, depth = render_rgbd(self.renderer, self.data, "onboard")
        geom_ids = render_geom_ids(self.renderer, self.data, "onboard")
        return ReferenceView(rgb, depth, self._lut[geom_ids])  # -1 indexes the trailing -1 entry

    def index(self, name: str) -> int:
        return self.names.index(name)
