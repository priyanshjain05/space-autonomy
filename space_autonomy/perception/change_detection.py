"""Model-based change detection: compare what the camera sees with what it should see.

For every pixel we have two depth readings (actual vs. expected) and two
colours. Three kinds of difference fall out of that:

    FARTHER  actual depth > expected   material that should be there is gone
    NEARER   actual depth < expected   something is there that shouldn't be
    COLOUR   same depth, different colour   the surface itself changed

and the mix of them says what kind of anomaly it is:

    mostly FARTHER over the component            -> missing
    FARTHER and NEARER around the component      -> misaligned (moved: vacated + new)
    NEARER only                                  -> foreign object
    COLOUR only                                  -> surface damage

No learning involved. It relies on knowing the robot's pose precisely
enough to line the two images up, which holds today (perfect state) and
will break once navigation is noisy. That breakage is the motivation for
learned perception later (M12).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..core.types import Observation
from .reference import ReferenceView

NOMINAL, MISSING, MISALIGNED, FOREIGN, DAMAGE, UNOBSERVABLE = (
    "nominal", "missing", "misaligned", "foreign_object", "surface_damage", "unobservable")


@dataclass(frozen=True)
class Thresholds:
    depth: float = 0.02        # m; depth sensor noise is 0.005 m, so this is 4 sigma
    color: float = 0.15        # max per-channel difference, 0..1
    min_px: int = 250          # smallest change region worth reporting, in pixels
    missing_frac: float = 0.6  # share of the component's pixels that must be gone to call it missing
    touch_px: int = 3          # new material must come within this many pixels of the component to be blamed on it


@dataclass
class Assessment:
    component: str
    kind: str
    confidence: float          # heuristic; the agent replaces it with a calibrated one when it has a model
    expected_px: int
    farther_px: int
    nearer_px: int
    color_px: int
    # Shape of the change, which tells real faults from misalignment (see _shape):
    edge_frac: float = 0.0     # share of changed pixels lying on edges the twin predicts
    thickness_px: float = 0.0  # half-width of the thickest changed region
    masks: dict = field(default_factory=dict, repr=False)  # for visualisation


def _open(mask):
    """Morphological opening: erase specks smaller than 3x3, keep larger regions intact."""
    return ndimage.binary_opening(mask, structure=np.ones((3, 3)))


def _touching(mask, target, reach):
    """The connected regions of `mask` that come within `reach` pixels of `target`.

    This is how change gets blamed on the right component. A tilted solar
    panel can poke into the antenna's camera view; that new material isn't
    connected to the antenna, so it isn't the antenna's anomaly.
    """
    regions, n = ndimage.label(mask)
    if n == 0:
        return mask
    near_target = ndimage.binary_dilation(target, iterations=reach)
    hit = np.unique(regions[near_target & mask])
    return np.isin(regions, hit[hit > 0])


def _shape(change, ref_depth, edge_jump=0.05):
    """How the changed pixels are laid out.

    Misregistration produces thin strips hugging the twin's depth edges; a real
    fault produces a solid blob. So: what share of the change sits on predicted
    edges, and how thick (in pixels, from the middle to the border) its thickest
    part is.
    """
    total = int(change.sum())
    if total == 0:
        return 0.0, 0.0
    d = ref_depth.astype(float)
    edges = np.zeros_like(change)
    edges[:, 1:] |= np.abs(np.diff(d, axis=1)) > edge_jump
    edges[1:, :] |= np.abs(np.diff(d, axis=0)) > edge_jump
    edges = ndimage.binary_dilation(edges, iterations=2)
    return float((change & edges).sum() / total), float(ndimage.distance_transform_edt(change).max())


def _confidence(px, min_px):
    """0.5 at the detection threshold, rising toward 1 as the evidence grows."""
    return float(1 - 0.5 ** (px / min_px))


def assess(obs: Observation, ref: ReferenceView, component: int, name: str, th: Thresholds = Thresholds()) -> Assessment:
    expected = ref.labels == component
    n_exp = int(expected.sum())
    if n_exp < th.min_px:
        return Assessment(name, UNOBSERVABLE, 0.0, n_exp, 0, 0, 0)

    d = obs.depth.astype(float) - ref.depth.astype(float)
    farther = _open(d > th.depth)
    nearer = _open(d < -th.depth)
    blur = lambda img: ndimage.uniform_filter(img / 255.0, size=(3, 3, 1))  # average out sensor noise
    color_diff = np.abs(blur(obs.rgb) - blur(ref.rgb)).max(axis=2)
    recolored = _open((color_diff > th.color) & ~farther & ~nearer)

    new_material = _touching(nearer, expected, th.touch_px)
    far_px = int((farther & expected).sum())
    near_px = int(new_material.sum())
    color_px = int((recolored & expected).sum())
    masks = {"expected": expected, "farther": farther & expected, "nearer": new_material,
             "color": recolored & expected}
    shape = _shape(masks["farther"] | masks["nearer"] | masks["color"], ref.depth)

    if far_px >= th.missing_frac * n_exp and near_px < th.min_px:
        kind, px = MISSING, far_px
    elif far_px >= th.min_px and near_px >= th.min_px:
        kind, px = MISALIGNED, far_px + near_px
    elif near_px >= th.min_px:
        kind, px = FOREIGN, near_px
    elif color_px >= th.min_px:
        kind, px = DAMAGE, color_px
    else:
        biggest = max(far_px, near_px, color_px)
        return Assessment(name, NOMINAL, 1 - 0.5 * biggest / th.min_px, n_exp, far_px, near_px, color_px,
                          *shape, masks)
    return Assessment(name, kind, _confidence(px, th.min_px), n_exp, far_px, near_px, color_px, *shape, masks)
