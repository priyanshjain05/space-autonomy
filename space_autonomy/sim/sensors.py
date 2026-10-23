"""Onboard sensors as the autonomy sees them: rendered by the simulator, then made imperfect."""

import mujoco
import numpy as np

from ..core.types import Observation
from .render import render_rgbd


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
