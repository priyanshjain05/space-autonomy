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
