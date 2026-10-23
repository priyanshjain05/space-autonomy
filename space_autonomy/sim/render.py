"""Offscreen rendering, plus debug overlays (flown trail, planned waypoints)."""

import mujoco
import numpy as np


def close_renderer(renderer):
    """Free a MuJoCo renderer's GPU resources inside its *own* OpenGL context.

    mujoco.Renderer.close() (also run when it's garbage-collected) frees its
    framebuffers in whatever context is current. With several renderers alive,
    that can be another renderer's context, whose buffers then get deleted
    instead and it renders garbage. Making the owner's context current first
    avoids that. (Reaches into two private attributes of mujoco.Renderer.)
    """
    ctx = getattr(renderer, "_gl_context", None)
    if ctx is not None and getattr(renderer, "_mjr_context", None) is not None:
        ctx.make_current()
    renderer.close()


class SceneRenderer:
    def __init__(self, sim, width=640, height=480):
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model, height, width, max_geom=20000)

    def render(self, camera, trail=None, waypoints=None, extra_trails=()) -> np.ndarray:
        """RGB image from a named camera, or a mujoco.MjvCamera. Overlays are drawn only when given.

        extra_trails: [(points, rgba), ...] more polylines, e.g. a piece of debris's path.
        """
        self.renderer.update_scene(self.sim.data, camera=camera)
        scene = self.renderer.scene
        for pts, rgba in ([(trail, (1.0, 0.6, 0.2, 1.0))] if trail is not None else []) + list(extra_trails):
            for a, b in zip(pts[:-1], pts[1:]):
                _add_line(scene, a, b, rgba)
        for p in waypoints if waypoints is not None else []:
            _add_sphere(scene, p, 0.06, (0.2, 1.0, 0.4, 0.9))
        return self.renderer.render()

    def close(self):
        close_renderer(self.renderer)

    def __del__(self):
        self.close()


def free_camera(lookat, distance, azimuth, elevation):
    """A camera aimed at a point from a distance and direction (degrees), not attached to anything."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


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
