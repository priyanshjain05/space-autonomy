"""MuJoCo world: the inspector robot, optionally a target spacecraft, and nothing else.

This is the only module (with render.py) that imports MuJoCo. Everything
crossing the boundary is a RobotState going out or a Wrench coming in.
"""

import mujoco
import numpy as np

from ..core.rotations import look_at_quat
from ..core.types import RobotState, VehicleSpec, Wrench
from .spacecraft import Spacecraft
from .thrusters import ThrusterModel


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _camera_xyaxes(pos, target):
    """MuJoCo cameras look down their own -z axis; build the x/y axes for that."""
    forward = np.asarray(target, float) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return _fmt(np.concatenate([right, up]))


SUN_DIR = np.array([-0.4, 0.5, -0.75])  # direction sunlight travels, spacecraft frame


def build_mjcf(vehicle: VehicleSpec, spacecraft: Spacecraft | None, start_pos, start_quat, timestep,
               sun_dir=SUN_DIR) -> str:
    h = vehicle.half_size
    overview_pos = np.array([10.0, -10.0, 7.0])
    return f"""
<mujoco model="space_inspection">
  <!-- No gravity: in the target's local frame over a few minutes, orbital
       effects are small, so we ignore them for now (see README). -->
  <option timestep="{timestep}" gravity="0 0 0" integrator="implicitfast"/>
  <!-- Fixed scene size, so camera clipping planes (and so depth readings of
       empty space) don't change when a component is added or removed. -->
  <statistic center="0 0 0" extent="8"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <headlight ambient="0.12 0.12 0.14" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="stars" type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0"
             mark="random" markrgb="0.9 0.9 1" random="0.004" width="1024" height="1024"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" dir="{_fmt(sun_dir)}" diffuse="1 0.97 0.9" specular="0.3 0.3 0.3"/>
    <camera name="overview" pos="{_fmt(overview_pos)}" xyaxes="{_camera_xyaxes(overview_pos, [0, 0, 0])}" fovy="50"/>
    {spacecraft.mjcf() if spacecraft else ""}
    <body name="inspector" pos="{_fmt(start_pos)}" quat="{_fmt(start_quat)}">
      <freejoint name="inspector"/>
      <geom name="inspector_hull" type="box" size="{h} {h} {h}" mass="{vehicle.mass}" rgba="1 0.45 0.1 1"/>
      <geom name="inspector_lens" type="cylinder" size="0.05 0.02" pos="{h + 0.02} 0 0" zaxis="1 0 0"
            rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0" mass="0"/>
      <!-- Body +x is where the robot "looks". Camera looks along its -z,
           so camera z = -body x, camera y (image up) = body z. -->
      <camera name="onboard" pos="{h + 0.05} 0 0" xyaxes="0 -1 0 0 0 1" fovy="{vehicle.camera.fovy}"/>
    </body>
  </worldbody>
</mujoco>
"""


class SpaceSim:
    def __init__(self, vehicle: VehicleSpec, spacecraft: Spacecraft | None = None,
                 start_pos=(0.0, 0.0, 0.0), start_quat=None, timestep=0.01, sun_dir=SUN_DIR):
        start_pos = np.asarray(start_pos, float)
        if start_quat is None:  # by default, start facing the spacecraft
            start_quat = look_at_quat(-start_pos) if np.linalg.norm(start_pos) > 0 else np.array([1.0, 0, 0, 0])
        self.vehicle = vehicle
        self.spacecraft = spacecraft
        self.xml = build_mjcf(vehicle, spacecraft, start_pos, start_quat, timestep, sun_dir)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.thrusters = ThrusterModel(vehicle)
        self.last_applied = Wrench.zero()
        self._body = self.model.body("inspector").id
        self._hull = self.model.geom("inspector_hull").id
        sc = self.model.body("spacecraft").id if spacecraft else -1
        self._sc_geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == sc]

    @property
    def t(self) -> float:
        return float(self.data.time)

    def state(self) -> RobotState:
        # Free joint layout: qpos = [x y z qw qx qy qz], qvel = [vx vy vz wx wy wz].
        # MuJoCo reports free-joint linear velocity in the world frame and
        # angular velocity in the body frame.
        q, v = self.data.qpos, self.data.qvel
        return RobotState(self.t, q[0:3].copy(), q[3:7].copy(), v[0:3].copy(), v[3:6].copy())

    def step(self, cmd: Wrench, duration: float) -> bool:
        """Hold `cmd` for `duration` seconds (zero-order hold). Returns True if the hull touched anything."""
        touched = False
        n = max(1, round(duration / self.model.opt.timestep))
        for _ in range(n):
            applied = self.thrusters.apply(cmd, self.model.opt.timestep)
            R = self.data.xmat[self._body].reshape(3, 3)
            # xfrc_applied is in the world frame; thrusters push in the body frame.
            self.data.xfrc_applied[self._body, :3] = R @ applied.force
            self.data.xfrc_applied[self._body, 3:] = R @ applied.torque
            mujoco.mj_step(self.model, self.data)
            touched |= self.in_contact()
        self.last_applied = applied
        return touched

    def in_contact(self) -> bool:
        c = self.data.contact[: self.data.ncon]
        return bool(np.any((c.geom1 == self._hull) | (c.geom2 == self._hull)))

    def clearance(self, max_dist=20.0) -> float:
        """Closest hull-to-spacecraft surface distance in metres (inf if no spacecraft)."""
        if not self._sc_geoms:
            return float("inf")
        return min(mujoco.mj_geomDistance(self.model, self.data, self._hull, g, max_dist, None)
                   for g in self._sc_geoms)
