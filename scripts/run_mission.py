"""M2/M5: the inspector flies a fixed route around Alpha-03 and photographs each component.

    python scripts/run_mission.py                # headless run, saves report + images
    python scripts/run_mission.py --gif          # also record an animated GIF
    mjpython scripts/run_mission.py --viewer     # watch it live (macOS needs mjpython)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from space_autonomy.evaluation.harness import run_mission
from space_autonomy.evaluation.plots import plot_states, plot_trajectory
from space_autonomy.evaluation.report import format_report, save_run
from space_autonomy.planning.inspection_planner import fixed_route_inspector
from space_autonomy.scenarios.alpha03 import nominal
from space_autonomy.sim.render import SceneRenderer
from space_autonomy.sim.world import SpaceSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alpha03-nominal")
    ap.add_argument("--gif", action="store_true", help="record results/.../mission.gif")
    ap.add_argument("--viewer", action="store_true", help="live 3D viewer (run with mjpython on macOS)")
    ap.add_argument("--speed", type=float, default=20.0, help="viewer playback speed vs. real time")
    args = ap.parse_args()

    sc = nominal()
    sim = SpaceSim(sc.vehicle, sc.spacecraft, start_pos=sc.home)
    autonomy = fixed_route_inspector(sc.vehicle, sc.spacecraft.targets(), sc.spacecraft.obstacles(),
                                     sc.home, order=sc.route)
    out = Path(args.out)
    (out / "views").mkdir(parents=True, exist_ok=True)

    planned = [p for leg in autonomy.legs for p in leg.waypoints]
    cam = SceneRenderer(sim, 640, 480)
    frames, trail = [], []

    def on_event(ev, sim):
        # Arrival at an inspection viewpoint: save what the onboard camera sees.
        if ev["event"] == "arrived" and ev["leg"] != "home":
            n = sum(1 for _ in (out / "views").glob("*.png")) + 1
            plt.imsave(out / "views" / f"{n:02d}_{ev['leg']}.png", cam.render("onboard"))

    def on_tick(sim, rows):
        r = rows[-1]
        if len(rows) % 5 == 0:
            trail.append((r["x"], r["y"], r["z"]))
        if args.gif and len(rows) % 20 == 0:  # one frame per 2 s of mission time
            frames.append(np.hstack([gif_cam.render("overview", trail, planned), gif_cam.render("onboard")]))
        if viewer is not None:
            if not viewer.is_running():
                sys.exit(0)
            viewer.sync()
            time.sleep(max(0.0, 0.1 / args.speed))

    gif_cam = SceneRenderer(sim, 400, 300) if args.gif else None
    viewer = _open_viewer(sim) if args.viewer else None

    result = run_mission(sim, autonomy, max_time=sc.max_time, expected_targets=sc.route,
                         on_event=on_event, on_tick=on_tick, name=sc.name)

    save_run(out, result)
    plt.imsave(out / "overview.png", SceneRenderer(sim, 1280, 960).render("overview", trail, planned))
    plot_states(result.telemetry, out / "states.png", f"{sc.name}: fixed-route baseline")
    plot_trajectory(result.telemetry, out / "trajectory.png", sc.spacecraft.obstacles(),
                    [(t.name, t.viewpoint) for t in sc.spacecraft.targets()], f"{sc.name}: flown path")
    if frames:
        _save_gif(frames, out / "mission.gif")

    for e in result.events:
        print(f"t={e['t']:7.1f}s  {e['event']:<17} {e['leg']}")
    print()
    print(format_report(result.summary))
    print(f"\nOutputs in {out}/")
    if viewer is not None:
        viewer.close()


def _open_viewer(sim):
    import mujoco.viewer
    try:
        return mujoco.viewer.launch_passive(sim.model, sim.data)
    except RuntimeError as e:
        sys.exit(f"{e}\nOn macOS the live viewer needs:  mjpython scripts/run_mission.py --viewer")


def _save_gif(frames, path):
    from PIL import Image
    imgs = [Image.fromarray(f).quantize(colors=128) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)


if __name__ == "__main__":
    main()
