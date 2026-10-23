"""Pose refinement: correct the robot's pose estimate using the spacecraft itself, before comparing.

The noise sweep showed the twin comparison breaks at millimetres and tenths
of a degree of navigation error, because the two images no longer line up.
But the depth image and the twin's depth are the same 3D surface seen from
two slightly different places. ICP (iterative closest point) finds the small
rigid motion between the two point clouds, and that motion *is* the
navigation error as seen by the camera. Re-render the twin at the corrected
pose and the images line up again.

We use the point-to-plane variant: each observed point is pulled onto the
*plane* of its matching twin surface, not onto a specific twin point.
Spacecraft are mostly flat faces, and flat faces pin down distance along
their normal precisely while saying nothing about sliding sideways.
Point-to-point ICP gets stuck on that sliding; point-to-plane doesn't.

The danger is that alignment explains away a real fault (rotating the camera
to match a tilted panel, or sliding to fill the gap a missing part left).
Guards:
- trimming: the worst-matching 20% of points are ignored each round;
- coarse-to-fine: matches may be 15 cm apart in the first rounds but only
  2 cm by the end, so only surfaces that really fit decide the final answer;
- a plausibility bound: corrections larger than the navigation system could
  really be off by are rejected and the original estimate is kept.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..core.rotations import quat_angle, quat_from_axis_angle, quat_to_mat, mat_to_quat


@dataclass(frozen=True)
class RegistrationConfig:
    points: int = 6000              # observed points sampled per fit
    iterations: int = 30
    trim: float = 0.8               # fraction of best matches kept each iteration
    gate: float = 0.15              # m, match distance allowed at first ...
    final_gate: float = 0.02        # m, ... tightening to this, so stragglers (a missing part's gap) stop pulling
    edge_jump: float = 0.03         # m, twin pixels next to a depth jump this big are not used (bad normals)
    damping: float = 1e-4           # keeps directions the view can't see (e.g. sliding along a panel) near zero
    max_shift: float = 0.10         # m, larger corrections are implausible -> rejected
    max_rotation_deg: float = 3.0
    max_range: float = 20.0         # m, depth beyond this is empty space
    seed: int = 0


@dataclass(frozen=True)
class Registration:
    accepted: bool
    shift_m: float          # size of the camera position correction
    rotation_deg: float     # size of the camera rotation correction
    rms_before_m: float     # point-to-surface misfit before and after
    rms_after_m: float
    matched_points: int


def _grid(depth, fovy_deg, max_range):
    """Per-pixel 3D points in the camera frame (MuJoCo convention: looks along -z, y up)."""
    h, w = depth.shape
    f = (h / 2) / np.tan(np.radians(fovy_deg) / 2)
    v, u = np.mgrid[0:h, 0:w]
    d = depth.astype(float)
    P = np.stack([(u + 0.5 - w / 2) / f * d, -(v + 0.5 - h / 2) / f * d, -d], axis=-1)
    return P, d < max_range


def backproject(depth, fovy_deg, max_range):
    P, valid = _grid(depth, fovy_deg, max_range)
    return P[valid]


def surface_points(depth, fovy_deg, max_range, edge_jump):
    """Points with surface normals, skipping pixels on depth edges (their normals are meaningless)."""
    P, valid = _grid(depth, fovy_deg, max_range)
    d = depth.astype(float)
    ok = valid.copy()
    ok[1:-1, 1:-1] &= (valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:])
    ok[1:-1, 1:-1] &= np.maximum.reduce([np.abs(d[:-2, 1:-1] - d[2:, 1:-1]),
                                         np.abs(d[1:-1, :-2] - d[1:-1, 2:])]) < edge_jump
    ok[[0, -1], :] = ok[:, [0, -1]] = False
    n = np.zeros_like(P)
    n[1:-1, 1:-1] = np.cross(P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1])
    norm = np.linalg.norm(n, axis=-1)
    ok &= norm > 1e-12
    n[ok] /= norm[ok][:, None]
    flip = np.einsum("ij,ij->i", n[ok], P[ok]) > 0  # make normals face the camera
    normals = n[ok]
    normals[flip] *= -1
    return P[ok], normals


def icp(observed, expected, normals, cfg: RegistrationConfig):
    """Rigid motion (R, t) taking observed camera-frame points onto the expected surface."""
    rng = np.random.default_rng(cfg.seed)
    src = observed[rng.choice(len(observed), min(cfg.points, len(observed)), replace=False)]
    tree = cKDTree(expected)
    R, t = np.eye(3), np.zeros(3)

    def residuals(R, t):
        moved = src @ R.T + t
        dist, idx = tree.query(moved, distance_upper_bound=cfg.gate)
        found = np.isfinite(dist)
        r = np.full(len(src), cfg.gate)
        r[found] = np.einsum("ij,ij->i", moved[found] - expected[idx[found]], normals[idx[found]])
        return moved, idx, found, r

    _, _, _, r0 = residuals(R, t)
    rms_before = float(np.sqrt(np.mean(np.minimum(np.abs(r0), cfg.gate) ** 2)))
    for k in range(cfg.iterations):
        gate = max(cfg.final_gate, cfg.gate * (cfg.final_gate / cfg.gate) ** (2 * k / cfg.iterations))
        moved, idx, found, r = residuals(R, t)
        found &= np.abs(r) <= gate
        if found.sum() < 100:
            break
        keep = found & (np.abs(r) <= np.quantile(np.abs(r[found]), cfg.trim))
        p, n = moved[keep], normals[idx[keep]]
        # Linearised: moving p by a small rotation w and shift dt changes its plane distance by
        # w . (p x n) + dt . n. Solve for the (w, dt) that cancels the residuals, least squares.
        A = np.hstack([np.cross(p, n), n])
        b = -r[keep]
        x = np.linalg.solve(np.einsum("ni,nj->ij", A, A) + cfg.damping * len(p) * np.eye(6), np.einsum("ni,n->i", A, b))
        w, dt = x[:3], x[3:]
        angle = np.linalg.norm(w)
        dR = quat_to_mat(quat_from_axis_angle(w, angle)) if angle > 0 else np.eye(3)
        R, t = dR @ R, dR @ t + dt
        if np.linalg.norm(dt) < 1e-6 and angle < 1e-7 and gate == cfg.final_gate:
            break
    _, _, found, r = residuals(R, t)
    rms_after = float(np.sqrt(np.mean(np.minimum(np.abs(r), cfg.gate) ** 2)))
    return R, t, rms_before, rms_after, int(found.sum())


def refine_camera_pose(obs_depth, ref_depth, cam_R, cam_t, fovy_deg, cfg: RegistrationConfig = RegistrationConfig()):
    """Corrected world camera pose (R, t) and what happened.

    cam_R, cam_t: where the robot thinks its camera is (world frame).
    """
    observed = backproject(obs_depth, fovy_deg, cfg.max_range)
    expected, normals = surface_points(ref_depth, fovy_deg, cfg.max_range, cfg.edge_jump)
    if len(observed) < 200 or len(expected) < 200:
        return cam_R, cam_t, Registration(False, 0.0, 0.0, np.nan, np.nan, 0)

    # Observed points live in the true camera frame; expected ones in the estimated camera frame.
    # Q = R P + t maps true-camera coordinates into estimated-camera coordinates, so
    # true camera pose = estimated pose composed with (R, t).
    R, t, before, after, n = icp(observed, expected, normals, cfg)
    shift = float(np.linalg.norm(t))
    rot = float(np.degrees(quat_angle([1, 0, 0, 0], mat_to_quat(R))))
    ok = shift <= cfg.max_shift and rot <= cfg.max_rotation_deg and after <= before
    info = Registration(ok, shift, rot, before, after, n)
    if not ok:
        return cam_R, cam_t, info
    return cam_R @ R, cam_R @ t + cam_t, info


# ── Stage 2: image-plane alignment ──────────────────────────────────────────
#
# Depth can't see sliding along a flat surface or spinning about its normal,
# so after the 3D fit a solar panel that fills the view can still be a few
# pixels off. What *can* see that motion is the outline: the panel's edges
# against empty space, and the shading. Lucas-Kanade finds the image shift and
# rotation that line the twin's picture up with the camera's.


@dataclass(frozen=True)
class ImageAlignConfig:
    iterations: int = 15
    blur_px: float = 1.5            # smooth so the fit sees edges as slopes, not steps
    max_shift_px: float = 8.0       # larger corrections are implausible -> rejected
    max_rotation_deg: float = 1.5
    max_range: float = 20.0


@dataclass(frozen=True)
class ImageAlignment:
    accepted: bool
    dx_px: float
    dy_px: float
    rotation_deg: float


def _signal(rgb, depth, blur, max_range):
    """What to align: brightness, plus the silhouette (where there's any surface at all)."""
    gray = rgb.astype(float).mean(axis=2) / 255.0
    return ndimage.gaussian_filter(gray + (depth < max_range), blur)


def _warp(img, dx, dy, theta, order=1):
    """Sample img at the rotated (about the image centre) and shifted coordinates."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(float)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    c, s = np.cos(theta), np.sin(theta)
    xs = c * (x - cx) - s * (y - cy) + cx + dx
    ys = s * (x - cx) + c * (y - cy) + cy + dy
    if img.ndim == 2:
        return ndimage.map_coordinates(img, [ys, xs], order=order, mode="nearest")
    return np.stack([ndimage.map_coordinates(img[..., k], [ys, xs], order=order, mode="nearest")
                     for k in range(img.shape[2])], axis=-1)


def align_image(obs, ref, cfg: ImageAlignConfig = ImageAlignConfig()) -> ImageAlignment:
    """Shift + rotation (dx, dy, theta) such that ref sampled there matches obs."""
    target = _signal(obs.rgb, obs.depth, cfg.blur_px, cfg.max_range)
    source = _signal(ref.rgb, ref.depth, cfg.blur_px, cfg.max_range)
    gy, gx = np.gradient(source)
    h, w = source.shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    x, y = x - (w - 1) / 2, y - (h - 1) / 2
    p = np.zeros(3)  # dx, dy, theta
    for _ in range(cfg.iterations):
        warped = _warp(source, *p)
        wgx, wgy = _warp(gx, *p), _warp(gy, *p)
        r = (warped - target).ravel()
        # Robust weights: big residuals (an anomaly, say) count less than small ones.
        scale = 1.4826 * np.median(np.abs(r)) + 1e-6
        wts = 1.0 / np.maximum(1.0, np.abs(r) / (3 * scale))
        c, s = np.cos(p[2]), np.sin(p[2])
        J = np.column_stack([wgx.ravel(), wgy.ravel(),
                             (wgx * (-s * x - c * y) + wgy * (c * x - s * y)).ravel()])
        # einsum, not J.T @ ...: these are 300k-row products, and Accelerate's BLAS on macOS 14.4
        # has segfaulted inside exactly this call in long batch runs. The 3x3 sums don't need BLAS.
        Jw = J * wts[:, None]
        step = np.linalg.solve(np.einsum("ni,nj->ij", J, Jw) + 1e-9 * np.eye(3), -np.einsum("ni,n->i", Jw, r))
        p += step
        if np.abs(step[:2]).max() < 1e-3 and abs(step[2]) < 1e-5:
            break
    ok = np.hypot(p[0], p[1]) <= cfg.max_shift_px and abs(np.degrees(p[2])) <= cfg.max_rotation_deg
    return ImageAlignment(bool(ok), float(p[0]), float(p[1]), float(np.degrees(p[2])))


def warp_view(ref, a: ImageAlignment):
    """The twin's images moved by an accepted image alignment."""
    if not a.accepted:
        return ref
    args = (a.dx_px, a.dy_px, np.radians(a.rotation_deg))
    return replace(ref,
                   rgb=np.clip(_warp(ref.rgb.astype(float), *args), 0, 255).astype(np.uint8),
                   depth=_warp_depth(ref.depth, *args),
                   labels=_warp(ref.labels, *args, order=0))


def _warp_depth(depth, dx, dy, theta):
    # Nearest-neighbour: interpolating across a depth edge would invent surfaces halfway between.
    return _warp(depth, dx, dy, theta, order=0).astype(np.float32)
