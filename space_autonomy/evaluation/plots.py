"""Matplotlib plots of a mission's telemetry."""

import itertools

import matplotlib

matplotlib.use("Agg")  # write files, never open windows
import matplotlib.pyplot as plt
import numpy as np


def _col(rows, key):
    return np.array([r[key] for r in rows], float)


def plot_states(rows, path, title=""):
    """Position, tracking error and propellant over time."""
    t = _col(rows, "t")
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for k in "xyz":
        ax[0].plot(t, _col(rows, k), label=k)
    ax[0].set_ylabel("position [m]")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, _col(rows, "pos_err_m"))
    ax[1].set_ylabel("distance to\ncurrent waypoint [m]")
    ax[2].plot(t, _col(rows, "att_err_deg"), color="tab:purple")
    ax[2].set_ylabel("pointing error [deg]")
    ax[3].plot(t, 1000 * _col(rows, "propellant_kg"), color="tab:red")
    ax[3].set_ylabel("propellant [g]")
    ax[3].set_xlabel("time [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


CHANGE_COLORS = {"farther": (0.25, 0.55, 1.0), "nearer": (1.0, 0.25, 0.2), "color": (1.0, 0.85, 0.1)}


def plot_assessment(obs, ref, a, path):
    """Onboard image | digital-twin image | where they differ, and what that means."""
    gray = (obs.rgb.mean(axis=2, keepdims=True) / 255.0).repeat(3, axis=2) * 0.5
    overlay = gray.copy()
    for key, rgb in CHANGE_COLORS.items():
        if key in a.masks:
            overlay[a.masks[key]] = rgb
    if "expected" in a.masks:  # outline where the component should be
        e = a.masks["expected"]
        edge = e & ~(np.roll(e, 1, 0) & np.roll(e, -1, 0) & np.roll(e, 1, 1) & np.roll(e, -1, 1))
        overlay[edge] = (0.2, 1.0, 0.4)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].imshow(obs.rgb)
    ax[0].set_title("Onboard camera (what is there)")
    ax[1].imshow(ref.rgb)
    ax[1].set_title("Digital twin (what should be there)")
    ax[2].imshow(overlay)
    verdict = a.kind.replace("_", " ").upper()
    ax[2].set_title(f"{a.component}: {verdict}  ({a.confidence:.0%})",
                    color="black" if a.kind == "nominal" else "firebrick", fontweight="bold")
    for x in ax:
        x.axis("off")
    from matplotlib.patches import Patch
    handles = [Patch(color=(0.2, 1.0, 0.4), label="expected outline")] + [
        Patch(color=c, label={"farther": "farther (missing material)", "nearer": "nearer (new material)",
                              "color": "colour change"}[k]) for k, c in CHANGE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _box_edges(box):
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    corners = [box.center + box.rotation @ (s * box.half) for s in signs]
    for i, j in itertools.combinations(range(8), 2):
        if np.abs(signs[i] - signs[j]).sum() == 2:  # corners differing in exactly one axis
            yield corners[i], corners[j]


def plot_trajectory(rows, path, obstacles=(), viewpoints=(), title=""):
    """3D path flown, with the spacecraft's obstacle boxes and the planned viewpoints."""
    p = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    for b in obstacles:
        for a, c in _box_edges(b):
            ax.plot(*zip(a, c), color="goldenrod", lw=0.8)
    ax.plot(*p.T, color="tab:orange", lw=1.5, label="flown path")
    ax.scatter(*p[0], color="black", s=40, label="start")
    for name, vp in viewpoints:
        ax.scatter(*vp, color="tab:green", s=30)
        ax.text(*vp, f" {name}", fontsize=7)
    lim = np.abs(np.vstack([p] + [b.center for b in obstacles] if obstacles else [p])).max() + 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
