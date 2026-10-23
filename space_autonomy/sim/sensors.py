"""Onboard sensors as the autonomy sees them: rendered by the simulator, then made imperfect."""

from dataclasses import replace

import mujoco
import numpy as np

from ..core.rotations import quat_from_axis_angle, quat_mul
from ..core.types import Observation, RobotState
from .render import render_rgbd


class NavigationError:
    """The gap between where the robot *thinks* it is and where it really is.

    Real relative navigation (cameras, star trackers, IMUs) doesn't jitter
    randomly every step; it drifts slowly. So each error component is a
    Gauss-Markov process: it wanders with a given spread (sigma) and forgets
    its past over a time constant (tau): errors 4 s apart are still ~88%
    correlated with tau = 30 s. Whatever the error is at the moment a picture
    is taken is what misaligns the digital twin.

    pos_sigma  m, per axis
    att_sigma  rad, per axis (a small rotation in the body frame)
    """

    def __init__(self, pos_sigma=0.0, att_sigma=0.0, tau=30.0, seed=0):
        self.pos_sigma, self.att_sigma, self.tau = pos_sigma, att_sigma, tau
        self.rng = np.random.default_rng(seed)
        self.pos_err = self.rng.normal(0, 1, 3) * pos_sigma
        self.att_err = self.rng.normal(0, 1, 3) * att_sigma
        self._t = None

    def estimate(self, s: RobotState) -> RobotState:
        if self._t is not None and s.t > self._t:
            a = np.exp(-(s.t - self._t) / self.tau)
            b = np.sqrt(1 - a * a)
            self.pos_err = a * self.pos_err + b * self.rng.normal(0, 1, 3) * self.pos_sigma
            self.att_err = a * self.att_err + b * self.rng.normal(0, 1, 3) * self.att_sigma
        self._t = s.t
        angle = np.linalg.norm(self.att_err)
        dq = quat_from_axis_angle(self.att_err, angle) if angle > 0 else np.array([1.0, 0, 0, 0])
        return replace(s, position=s.position + self.pos_err, quat=quat_mul(s.quat, dq))


class OnboardCamera:
    """RGB-D camera on the inspector's +x face.

    NASA's Astrobee carries the same pairing: a colour camera plus a
    time-of-flight depth camera. Noise here is simple per-pixel Gaussian;
    blur, exposure and lens effects can be added later.
    """

    def __init__(self, sim, rgb_noise=0.02, depth_noise=0.005, seed=0):
        cam = sim.vehicle.camera
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model, cam.height, cam.width)
        self.rgb_noise = rgb_noise        # std dev, fraction of full scale
        self.depth_noise = depth_noise    # std dev, metres
        self.rng = np.random.default_rng(seed)

    def capture(self) -> Observation:
        rgb, depth = render_rgbd(self.renderer, self.sim.data, "onboard")
        rgb = rgb / 255.0 + self.rng.normal(0, self.rgb_noise, rgb.shape)
        depth = depth + self.rng.normal(0, self.depth_noise, depth.shape).astype(np.float32)
        return Observation((np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth)
