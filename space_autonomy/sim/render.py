"""Offscreen rendering, plus debug overlays (flown trail, planned waypoints)."""

import mujoco
import numpy as np


class SceneRenderer:
    def __init__(self, sim, width=640, height=480):
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model, height, width, max_geom=20000)

    def render(self, camera: str, trail=None, waypoints=None) -> np.ndarray:
        """RGB image from a named camera. Overlays are drawn only when given."""
        self.renderer.update_scene(self.sim.data, camera=camera)
        scene = self.renderer.scene
        if trail is not None and len(trail) > 1:
            for a, b in zip(trail[:-1], trail[1:]):
                _add_line(scene, a, b, (1.0, 0.6, 0.2, 1.0))
        for p in waypoints if waypoints is not None else []:
            _add_sphere(scene, p, 0.06, (0.2, 1.0, 0.4, 0.9))
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def render_rgbd(renderer, data, camera):
    """(rgb uint8, depth float32 in metres) from a named camera."""
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def render_geom_ids(renderer, data, camera):
    """Which geom each pixel shows (-1 = empty space)."""
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    # The starfield skybox breaks segmentation output on macOS; space is "nothing" anyway.
    sky = mujoco.mjtRndFlag.mjRND_SKYBOX
    renderer.scene.flags[sky] = 0
    seg = renderer.render().copy()
    renderer.scene.flags[sky] = 1  # the flag outlives this call; put it back
    renderer.disable_segmentation_rendering()
    return np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)


def _next_geom(scene):
    if scene.ngeom >= scene.maxgeom:
        return None
    scene.ngeom += 1
    return scene.geoms[scene.ngeom - 1]


def _add_line(scene, a, b, rgba, width=2.0):
    g = _next_geom(scene)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))


def _add_sphere(scene, p, r, rgba):
    g = _next_geom(scene)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), np.asarray(p, float),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
