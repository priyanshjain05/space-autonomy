"""Loose debris: how it's sensed, and how the test harness aims it.

HazardSensor is what the robot gets: noisy positions of loose objects within
range, a few times a second, like a short-range lidar with a tracker. It
doesn't model occlusion or misdetections yet.

The injector is the adversary. It's allowed privileged knowledge the robot
never has: it replays the robot's trajectory from a clean run of the same
mission, picks a moment, and launches debris so that it arrives exactly
where the robot will be at that moment. Without avoidance, that's a hit.
"""

from dataclasses import dataclass

import numpy as np

from ..core.types import Detection
from ..planning.paths import segment_clear


class HazardSensor:
    def __init__(self, sim, range_m=None, pos_noise=0.02, rate_hz=5.0, seed=0):
        self.sim = sim
        self.range = sim.vehicle.hazard_range if range_m is None else range_m
        self.pos_noise = pos_noise
        self.period = 1.0 / rate_hz
        self.rng = np.random.default_rng(seed + 101)
        self._next = 0.0

    def detect(self) -> list[Detection]:
        if self.sim.t + 1e-9 < self._next:
            return []
        self._next = self.sim.t + self.period
        here = self.sim.state().position
        out = []
        for i, r in enumerate(self.sim.debris_radii):
            if not self.sim.launched[i]:
                continue
            p = self.sim.debris_position(i)
            if np.linalg.norm(p - here) - r <= self.range:
                out.append(Detection(self.sim.t, i, p + self.rng.normal(0, self.pos_noise, 3),
                                     r * (1 + self.rng.normal(0, 0.05))))
        return out


@dataclass(frozen=True)
class DebrisStrike:
    launch_t: float
    start: np.ndarray
    velocity: np.ndarray
    aim_t: float              # when it reaches the point where the robot would be
    aim_point: np.ndarray
    robot_moving: bool        # was the robot in transit at aim_t (vs. holding at a viewpoint)?


def plan_strike(rows, obstacles, speed, radius, rng, approach=8.0, earliest=40.0, tries=60) -> DebrisStrike:
    """Aim debris at the robot, using a clean run's telemetry rows (t, x, y, z, vx, vy, vz).

    The debris comes in across the robot's direction of travel (or from any direction if the
    robot is holding still), from up to `approach` metres away, along a line that doesn't pass
    through the spacecraft.
    """
    t = np.array([r["t"] for r in rows])
    pos = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    vel = np.array([[r["vx"], r["vy"], r["vz"]] for r in rows])
    candidates = np.nonzero((t >= earliest) & (t <= t[-1] - 20))[0]
    for _ in range(tries):
        k = rng.choice(candidates)
        aim_t, aim, v_robot = t[k], pos[k], vel[k]
        moving = np.linalg.norm(v_robot) > 0.05
        d = rng.normal(size=3)
        if moving:  # cross the robot's path rather than chase it or meet it head-on
            d -= (d @ v_robot) / (v_robot @ v_robot) * v_robot
        d /= np.linalg.norm(d)
        dist = min(approach, speed * (aim_t - 5.0))
        start = aim - d * dist
        if not segment_clear(start, aim + d * 2.0, obstacles, radius + 0.1):
            continue
        return DebrisStrike(aim_t - dist / speed, start, d * speed, aim_t, aim, bool(moving))
    raise RuntimeError("no clear approach line found for the debris")


class DebrisInjector:
    def __init__(self, strike: DebrisStrike, index=0):
        self.strike = strike
        self.index = index
        self.done = False

    def update(self, sim):
        if not self.done and sim.t + 1e-9 >= self.strike.launch_t:
            sim.launch_debris(self.index, self.strike.start, self.strike.velocity)
            self.done = True
