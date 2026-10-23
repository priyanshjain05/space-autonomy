"""Leg sequencer: the simplest possible mission executive.

A mission is a list of legs. Each leg is: fly through some waypoints,
stop at the last one, point the camera, hold still for a while, move on.
There is no decision-making here yet; the order is fixed in advance.
Smarter planners (M7+) will produce or rewrite legs on the fly.
"""

from dataclasses import dataclass, field

import numpy as np

from ..controllers.pose_controller import PoseController
from ..core.rotations import look_at_quat, quat_angle
from ..core.types import RobotState, Wrench


@dataclass
class Leg:
    name: str
    waypoints: list                   # pass-through points; the last one is the goal
    look_at: np.ndarray | None = None  # point the camera here at the goal ...
    goal_quat: np.ndarray | None = None  # ... or hold this exact attitude
    transit_look_at: np.ndarray | None = None  # what to face while travelling (None: goal attitude)
    dwell: float = 0.0                # s to hold at the goal (e.g. to take images)
    kind: str = "goto"                # "inspect" | "reinspect" | "goto" | "return"
    assessed: bool = False            # picture already taken here (a detour may bring the robot back)


@dataclass(frozen=True)
class Tolerances:
    position: float = 0.05     # m
    speed: float = 0.02        # m/s
    angle_deg: float = 3.0
    rate: float = 0.02         # rad/s
    pass_radius: float = 0.35  # m, how close counts as "passed" an intermediate waypoint


@dataclass
class LegSequencer:
    legs: list
    controller: PoseController
    tol: Tolerances = Tolerances()
    leg_index: int = 0
    wp_index: int = 0
    phase: str = "transit"       # transit -> settling -> dwell -> (next leg) ... -> complete
    _goal_quat: np.ndarray | None = None
    _dwell_until: float = 0.0
    _events: list = field(default_factory=list)
    _target_pos: np.ndarray | None = None
    _target_quat: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self.phase == "complete"

    @property
    def leg(self) -> Leg:
        return self.legs[min(self.leg_index, len(self.legs) - 1)]

    def status(self) -> dict:
        return {"leg": self.leg.name, "phase": self.phase,
                "target_pos": self._target_pos, "target_quat": self._target_quat}

    def pop_events(self) -> list:
        ev, self._events = self._events, []
        return ev

    def _emit(self, t, kind, **info):
        self._events.append({"t": round(t, 3), "event": kind, "leg": self.leg.name, **info})

    def command(self, s: RobotState) -> Wrench:
        leg = self.leg
        at_goal_wp = self.wp_index == len(leg.waypoints) - 1
        target = np.asarray(leg.waypoints[self.wp_index], float)

        if not at_goal_wp and np.linalg.norm(target - s.position) < self.tol.pass_radius:
            self.wp_index += 1
            return self.command(s)

        if at_goal_wp:
            if self._goal_quat is None:
                # Freeze the goal attitude once, keeping the current roll.
                self._goal_quat = (leg.goal_quat if leg.goal_quat is not None
                                   else look_at_quat(leg.look_at - target, s.rotation[:, 2]))
                if self.phase == "transit":
                    self.phase = "settling"
            quat = self._goal_quat
        else:
            facing = leg.transit_look_at if leg.transit_look_at is not None else leg.look_at
            quat = (look_at_quat(facing - s.position, s.rotation[:, 2]) if facing is not None
                    else leg.goal_quat)

        if self.phase == "settling" and self._settled(s, target, quat):
            self.phase = "dwell"
            self._dwell_until = s.t + leg.dwell
            self._emit(s.t, "arrived")
        if self.phase == "dwell" and s.t >= self._dwell_until:
            self._finish_leg(s)

        self._target_pos, self._target_quat = target, quat
        return self.controller.command(s, target, quat, cruise=not at_goal_wp)

    def reroute(self, waypoints) -> None:
        """Fly the current leg through new waypoints (same goal). If the robot was already at the goal,
        it has to come back and settle again."""
        self.leg.waypoints = list(waypoints)
        self.wp_index = 0
        if self.phase in ("settling", "dwell"):
            self.phase = "transit"
        self._goal_quat = None

    def detour(self, leg: Leg, replan_next) -> None:
        """Fly `leg` right after the current one, then carry on.

        The next leg's path was planned from where the robot is now, not from
        the end of the detour, so it is replaced: replan_next(old_leg, start)
        returns the same leg planned from `start`.
        """
        i = self.leg_index + 1
        after = [replan_next(self.legs[i], leg.waypoints[-1])] if i < len(self.legs) else []
        self.legs[i:i + 1] = [leg] + after

    def _settled(self, s, target, quat) -> bool:
        t = self.tol
        return (np.linalg.norm(target - s.position) < t.position
                and np.linalg.norm(s.velocity) < t.speed
                and np.degrees(quat_angle(s.quat, quat)) < t.angle_deg
                and np.linalg.norm(s.angular_velocity) < t.rate)

    def _finish_leg(self, s):
        if self.leg.kind == "inspect":
            self._emit(s.t, "inspected")
        if self.leg_index == len(self.legs) - 1:
            self.phase = "complete"
            self._emit(s.t, "mission_complete")
            return
        self.leg_index += 1
        self.wp_index = 0
        self.phase = "transit"
        self._goal_quat = None
