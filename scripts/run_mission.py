"""Inspection mission around Alpha-03: fly the route, look at every component, report anomalies.

    python scripts/run_mission.py                  # random anomalies (seed 1)
    python scripts/run_mission.py --seed 7         # a different random spacecraft
    python scripts/run_mission.py --clean          # undamaged spacecraft
    python scripts/run_mission.py --showcase       # one anomaly of each type
    python scripts/run_mission.py --gif            # also record an animated GIF
    mjpython scripts/run_mission.py --viewer       # watch it live (macOS needs mjpython)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from space_autonomy.autonomy.agent import make_inspection_agent
from space_autonomy.evaluation.anomaly_scoring import score_anomalies
from space_autonomy.evaluation.harness import run_mission
from space_autonomy.evaluation.plots import plot_assessment, plot_states, plot_trajectory
from space_autonomy.evaluation.report import format_report, save_run
from space_autonomy.scenarios.alpha03 import nominal, showcase, with_anomalies
from space_autonomy.sim.render import SceneRenderer
from space_autonomy.sim.sensors import OnboardCamera
from space_autonomy.sim.world import SpaceSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="which random set of anomalies")
    ap.add_argument("--clean", action="store_true", help="no anomalies")
    ap.add_argument("--showcase", action="store_true", help="one anomaly of each type")
    ap.add_argument("--out", default=None, help="output folder (default results/<scenario>)")
    ap.add_argument("--gif", action="store_true", help="record mission.gif")
    ap.add_argument("--viewer", action="store_true", help="live 3D viewer (run with mjpython on macOS)")
    ap.add_argument("--speed", type=float, default=20.0, help="viewer playback speed vs. real time")
    args = ap.parse_args()

    sc = nominal() if args.clean else showcase() if args.showcase else with_anomalies(args.seed)
    sim = SpaceSim(sc.vehicle, sc.world, start_pos=sc.home)
    agent = make_inspection_agent(sc.vehicle, sc.spacecraft, sc.home, sc.route)
    camera = OnboardCamera(sim, seed=sc.seed)
    out = Path(args.out or f"results/{sc.name}")
    (out / "views").mkdir(parents=True, exist_ok=True)
    for old in (out / "views").glob("*.png"):
        old.unlink()

    print(f"Scenario {sc.name}: {len(sc.anomalies)} hidden anomalies")
    planned = [p for leg in agent.legs for p in leg.waypoints]
    frames, trail = [], []

    def on_event(ev, sim):
        if ev["event"] == "assessed":
            n = len(list((out / "views").glob("*.png"))) + 1
            plot_assessment(*agent.last, out / "views" / f"{n:02d}_{ev['leg']}.png")

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
    if viewer is not None:
        print("Viewer open (it may be behind other windows: Cmd+Tab to 'mjpython'). Starting in 5 s...")
        _hold(viewer, 5.0)

    result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route, camera=camera,
                         on_event=on_event, on_tick=on_tick, name=sc.name)
    result.summary.update(score_anomalies(agent.world_model, sc.anomalies, sc.route))

    save_run(out, result)
    wm_text = agent.world_model.table(f"WORLD MODEL: {sc.spacecraft.name}")
    (out / "world_model.txt").write_text(wm_text + "\n")
    (out / "world_model.json").write_text(json.dumps(agent.world_model.to_dict(), indent=2))
    plt.imsave(out / "overview.png", SceneRenderer(sim, 1280, 960).render("overview", trail, planned))
    plot_states(result.telemetry, out / "states.png", f"{sc.name}: fixed-route baseline")
    plot_trajectory(result.telemetry, out / "trajectory.png", sc.spacecraft.obstacles(),
                    [(t.name, t.viewpoint) for t in sc.spacecraft.targets()], f"{sc.name}: flown path")
    if frames:
        _save_gif(frames, out / "mission.gif")

    for e in result.events:
        if e["event"] == "assessed":
            print(f"t={e['t']:7.1f}s  looked at {e['leg']:<14} -> {e['result']} ({e['confidence']:.0%})")
    print()
    print(wm_text)
    print()
    print("GROUND TRUTH")
    for d in result.summary["anomaly_details"] or [{"component": "-", "injected": "none", "detail": ""}]:
        print(f"  {d['component']:<16}{d['injected']:<17}{d['detail']}")
    print()
    print(format_report(result.summary))
    print(f"\nOutputs in {out}/")
    if viewer is not None:
        print("\nMission over. Close the viewer window (or Ctrl+C) to exit.")
        _hold(viewer, float("inf"))


def _hold(viewer, seconds):
    """Keep the viewer responsive without advancing the simulation."""
    end = time.time() + seconds
    while viewer.is_running() and time.time() < end:
        viewer.sync()
        time.sleep(0.03)


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
