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
