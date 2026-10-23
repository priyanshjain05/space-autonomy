"""One debris encounter, flown twice: ignoring the debris, then avoiding it.

    python scripts/run_hazard.py                  # seed 6001, debris at 0.15 m/s, 4 m sensor
    python scripts/run_hazard.py --seed 6004 --gif
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from space_autonomy.autonomy.agent import make_inspection_agent
from space_autonomy.evaluation.harness import run_mission
from space_autonomy.scenarios.alpha03 import with_anomalies
from space_autonomy.sim.hazards import DebrisInjector, HazardSensor, plan_strike
from space_autonomy.sim.render import SceneRenderer, free_camera
from space_autonomy.sim.sensors import OnboardCamera
from space_autonomy.sim.world import SpaceSim

DEBRIS_TRAIL = (0.6, 0.85, 1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6001)
    ap.add_argument("--speed", type=float, default=0.15, help="debris speed, m/s")
    ap.add_argument("--range", type=float, default=4.0, help="hazard sensor range, m")
    ap.add_argument("--radius", type=float, default=0.25, help="debris radius, m")
    ap.add_argument("--gif", action="store_true", help="film the encounter both ways, side by side")
    ap.add_argument("--out", default="results/hazard_demo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sc = with_anomalies(args.seed)
    clean = fly(sc, args, avoid=False)
    strike = plan_strike(clean[0].telemetry, sc.world.obstacles(), args.speed, args.radius,
                         np.random.default_rng(sc.seed + 7))
    print(f"Debris launched at t={strike.launch_t:.0f}s, aimed to hit the robot at t={strike.aim_t:.0f}s "
          f"({'in transit' if strike.robot_moving else 'holding at a viewpoint'})\n")

    films = {}
    for avoid in (False, True):
        result, frames = fly(sc, args, avoid, strike)
        films[avoid] = frames
        s = result.summary
        print(f"{'AVOID ' if avoid else 'IGNORE'}  debris hits {s['debris_collisions']}, closest gap "
              f"{s['min_debris_clearance_m']} m, mission {s['mission_time_s']:.0f} s "
              f"(+{s['mission_time_s'] - clean[0].summary['mission_time_s']:.0f}), propellant {s['propellant_used_pct']}%")
        for e in result.events:
            if e["event"] in ("hazard_detected", "replanned", "resumed", "no_safe_option"):
                info = {k: v for k, v in e.items() if k not in ("t", "event")}
                print(f"   t={e['t']:6.1f}s  {e['event']:<16} {info}")
        print()
    if args.gif:
        save_gif(films, out / "encounter.gif")
        print(f"Saved {out / 'encounter.gif'}")


def fly(sc, args, avoid, strike=None):
    vehicle = replace(sc.vehicle, hazard_range=args.range)
    sim = SpaceSim(vehicle, sc.world, start_pos=sc.home, debris_radii=(args.radius,) if strike else ())
    agent = make_inspection_agent(vehicle, sc.spacecraft, sc.home, sc.route, avoid=avoid)
    frames, robot_trail, debris_trail = [], [], []
    cam = SceneRenderer(sim, 480, 360) if (strike and args.gif) else None

    def on_tick(sim, rows):
        if cam is None or not (strike.launch_t - 2 <= sim.t <= strike.aim_t + 25):
            return
        robot_trail.append(sim.state().position)
        debris_trail.append(sim.debris_position(0))
        if len(rows) % 2 == 0:  # a frame per 0.4 s of mission time
            view = free_camera(strike.aim_point, 7.0, 150, -25)
            frames.append(cam.render(view, np.array(robot_trail), extra_trails=[(np.array(debris_trail), DEBRIS_TRAIL)]))

    result = run_mission(sim, agent, max_time=sc.max_time, expected_targets=sc.route,
                         camera=OnboardCamera(sim, seed=sc.seed),
                         hazard_sensor=HazardSensor(sim, seed=sc.seed) if strike else None,
                         injector=DebrisInjector(strike) if strike else None,
                         log_hz=5.0 if strike else 10.0, on_tick=on_tick, name=sc.name)
    return result, frames


def save_gif(films, path):
    from PIL import Image, ImageDraw
    n = min(len(films[False]), len(films[True]))
    imgs = []
    for k in range(n):
        frame = Image.fromarray(np.hstack([films[False][k], films[True][k]]))
        d = ImageDraw.Draw(frame)
        d.text((10, 8), "Ignores loose objects", fill=(255, 160, 90))
        d.text((490, 8), "Detects, stops, replans", fill=(120, 180, 255))
        imgs.append(frame.quantize(colors=128))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=70, loop=0, optimize=True)


if __name__ == "__main__":
    main()
