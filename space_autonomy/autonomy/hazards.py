"""Unexpected obstacles: detect, stop, re-observe, replan, resume (M8/M9).

    Obstacle detected
          ↓
    Stop and hold position
          ↓
    Re-observe (a couple of seconds, so the velocity estimate settles)
          ↓
    Predict: where will it be, where would I be on each candidate plan?
          ↓
    Candidates: wait here until it passes, or detour through a side point
          ↓
    Pick the cheapest one that keeps a safe gap the whole way
          ↓
    Execute, keep watching (re-plan if the prediction goes wrong)
          ↓
    Continue the mission

The object is predicted at constant velocity (it's drifting debris, nothing
pushes it); the robot by replaying its own controller along the plan.

If the conflict is only seconds away there's no time to stop and look: the
robot decides straight away with what it has ("urgent").
"""

from dataclasses import dataclass, field

import numpy as np

from ..core.types import Detection, RobotState
from ..planning.paths import segment_clear


# ── Tracking ────────────────────────────────────────────────────────────────


@dataclass
class Track:
    id: int
    radius: float
    times: list = field(default_factory=list)
    positions: list = field(default_factory=list)

    def add(self, d: Detection):
        self.times.append(d.t)
        self.positions.append(np.asarray(d.position, float))
        self.radius = 0.8 * self.radius + 0.2 * d.radius

    def estimate(self, window=4.0):
        """(position now, velocity) from a straight-line fit over the last `window` seconds."""
        t = np.array(self.times)
        keep = t >= t[-1] - window
        t, p = t[keep], np.array(self.positions)[keep]
        if len(t) < 2:
            return p[-1], np.zeros(3)
        A = np.column_stack([np.ones_like(t), t - t[-1]])
        coef, *_ = np.linalg.lstsq(A, p, rcond=None)
        return coef[0], coef[1]

    @property
    def last_seen(self):
        return self.times[-1]


class Tracker:
    def __init__(self):
        self.tracks: dict[int, Track] = {}

    def update(self, detections):
        for d in detections:
            self.tracks.setdefault(d.track_id, Track(d.track_id, d.radius)).add(d)

    def active(self, t, max_age=1.0):
        return [tr for tr in self.tracks.values() if t - tr.last_seen <= max_age]


# ── Prediction ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Motion:
    """How the robot really moves, for predicting it: the pose controller's own numbers."""
    max_speed: float = 0.25   # m/s, PoseGains.max_speed
    max_accel: float = 0.1    # m/s^2, max_force / mass
    pos_gain: float = 0.15    # 1/s, PoseGains.pos_gain: slows down into a stop
    vel_gain: float = 0.8     # 1/s, PoseGains.vel_gain


def timeline(start, waypoints, t0, *, v0=0.0, wait=0.0, pauses=None, motion: Motion = Motion(),
             horizon=60.0, dt=0.1):
    """Where the robot will be, as a function of time.

    Hold `wait` s, then fly the waypoints, stopping pauses[k] seconds at waypoint k (that's where it
    holds still to inspect) and at the last one. Rather than assume a constant speed, this replays the
    pose controller in one dimension along the path: it accelerates at most `max_accel`, cruises at
    `max_speed`, and slows into each stop the way the real controller does. From rest, a constant-speed
    guess gets the first few seconds wrong by a factor of two, which is exactly when dodging matters.
    v0 is the robot's current speed along the path.
    """
    pauses = pauses or {}
    pts = [np.asarray(start, float)] + [np.asarray(w, float) for w in waypoints]
    seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    stops = [(cum[k + 1], p) for k, p in sorted(pauses.items()) if p > 0 and k + 1 < len(cum)]
    stops.append((cum[-1], np.inf))

    m = motion
    t, s, v = t0, 0.0, max(0.0, v0)
    hold_until, k = t0 + wait, 0
    times, dists = [t], [s]
    while t < t0 + horizon:
        target, pause = stops[k]
        if t < hold_until:
            v_des = 0.0
        else:
            d = target - s
            v_des = min(m.max_speed, m.pos_gain * d)
            if d < 0.05 and v < 0.03 and np.isfinite(pause):  # settled at a stop: hold, then go on
                hold_until, k = t + pause, k + 1
        v = max(0.0, v + np.clip(m.vel_gain * (v_des - v), -m.max_accel, m.max_accel) * dt)
        s = min(s + v * dt, target)
        t += dt
        times.append(t)
        dists.append(s)
    times, dists = np.array(times), np.array(dists)
    xyz = np.column_stack([np.interp(dists, cum, np.array(pts)[:, i]) for i in range(3)])

    def at(t):
        i = np.searchsorted(times, t)
        if i <= 0:
            return xyz[0]
        if i >= len(times):
            return xyz[-1]
        f = (t - times[i - 1]) / (times[i] - times[i - 1])
        return xyz[i - 1] + f * (xyz[i] - xyz[i - 1])
    arrive = times[np.argmax(dists >= cum[-1] - 0.05)] if dists[-1] >= cum[-1] - 0.05 else np.inf
    return at, arrive


def min_gap(robot_at, obj_pos, obj_vel, gap_offset, t0, horizon, dt=0.25):
    """Smallest predicted surface-to-surface gap, and when (gap_offset = both radii)."""
    best, when = np.inf, t0
    for t in np.arange(t0, t0 + horizon, dt):
        g = np.linalg.norm(robot_at(t) - (obj_pos + obj_vel * (t - t0))) - gap_offset
        if g < best:
            best, when = g, t
    return best, when


# ── Response ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HazardConfig:
    safety: float = 0.5        # m, surface-to-surface gap the robot insists on
    horizon: float = 60.0      # s, how far ahead it predicts
    motion: Motion = Motion()  # how the robot moves, for predicting it
    observe: float = 2.0       # s, watch the object after stopping, before deciding ...
    urgent: float = 12.0       # s, ... unless the conflict is closer than this: then decide at once.
                               # (2 s of looking + ~2.5 s to brake + ~6 s to get clear from rest)
    rethink: float = 1.0       # s, when nothing is fully safe: hold the least-bad choice this long, then reconsider
    check_every: float = 0.5   # s
    min_obs: int = 4           # sightings needed before a track is trusted
    detour_dists: tuple = (0.6, 1.2, 2.0)  # m, how far to sidestep
    detour_dirs: int = 8


class HazardResponder:
    """Sits between the mission sequencer and the controller. Mostly it just watches.

    When a tracked object is predicted to come within `safety` of where the robot is about to be,
    it takes over: hold position, look, then either wait for the object to pass or reroute the
    current leg through a side point. Every decision is logged as an event.
    """

    def __init__(self, planner, robot_radius, cfg: HazardConfig = HazardConfig()):
        self.planner = planner
        self.robot_radius = robot_radius
        self.cfg = cfg
        self.tracker = Tracker()
        self.mode = "clear"          # clear | holding | waiting
        self.hold = None             # (position, quat) while holding/waiting
        self._since = 0.0
        self._resume_at = 0.0
        self._next_check = 0.0
        self.events = []

    def on_detections(self, dets):
        self.tracker.update(dets)

    def _emit(self, t, kind, **info):
        self.events.append({"t": round(t, 3), "event": kind, **info})

    def _planned(self, seq, t, rest=None):
        """What the sequencer will do next: waypoints from where the robot is, and where it will pause.

        Covers the rest of the current leg (or `rest` instead, for a candidate reroute), the hold at
        its goal, and the next leg.
        """
        if seq.done:
            return [], {}
        leg = seq.leg
        candidate = rest is not None
        rest = [np.asarray(w, float) for w in (rest if candidate else leg.waypoints[seq.wp_index:])]
        if seq.phase == "dwell" and not candidate:  # already holding at the goal: just what's left of it
            pause = max(0.0, seq._dwell_until - t)
        else:
            pause = leg.dwell + (5.0 if leg.dwell else 0.0)  # settling takes a few seconds too
        nxt = seq.legs[seq.leg_index + 1].waypoints if seq.leg_index + 1 < len(seq.legs) else []
        return rest + [np.asarray(w, float) for w in nxt], {len(rest) - 1: pause}

    def _worst(self, s: RobotState, plan, wait=0.0):
        """Worst predicted gap to any trusted track if the robot holds `wait` s then follows `plan`."""
        waypoints, pauses = plan
        at, _ = timeline(s.position, waypoints, s.t, v0=_speed_along(s, waypoints), wait=wait, pauses=pauses,
                         motion=self.cfg.motion, horizon=self.cfg.horizon)
        worst, who, when = np.inf, None, None
        for tr in self.tracker.active(s.t):
            if len(tr.times) < self.cfg.min_obs:
                continue
            p, v = tr.estimate()
            g, w = min_gap(at, p, v, self.robot_radius + tr.radius, s.t, self.cfg.horizon)
            if g < worst:
                worst, who, when = g, tr, w
        return worst, who, when

    def step(self, s: RobotState, seq):
        """None to let the sequencer fly, or (position, quat) to hold."""
        cfg = self.cfg
        if self.mode == "clear":
            if s.t < self._next_check:
                return None
            self._next_check = s.t + cfg.check_every
            gap, tr, when = self._worst(s, self._planned(seq, s.t))
            if gap < cfg.safety:
                urgent = when - s.t < cfg.urgent
                self.mode, self.hold, self._since = "holding", (s.position.copy(), s.quat.copy()), s.t
                self._emit(s.t, "hazard_detected", track=tr.id, predicted_gap_m=round(gap, 3),
                           seconds_to_conflict=round(when - s.t, 1), urgent=urgent, leg=seq.leg.name)
                if urgent:  # no time to stop and look
                    self._decide(s, seq)
                return self.hold if self.mode != "clear" else None
            return None

        if self.mode == "holding":
            if s.t - self._since >= cfg.observe and np.linalg.norm(s.velocity) < 0.05:
                self._decide(s, seq)
            return self.hold if self.mode != "clear" else None

        if self.mode == "waiting":
            if s.t >= self._resume_at:
                gap, _, _ = self._worst(s, self._planned(seq, s.t))
                if gap >= cfg.safety:
                    self.mode = "clear"
                    self._emit(s.t, "resumed", leg=seq.leg.name)
                    return None
                self._decide(s, seq)  # still in the way: think again
            return self.hold if self.mode != "clear" else None
        return None

    def _arrival(self, s: RobotState, waypoints) -> float:
        """Predicted seconds until the robot settles at the end of `waypoints`."""
        _, arrive = timeline(s.position, waypoints, s.t, v0=_speed_along(s, waypoints), motion=self.cfg.motion,
                             horizon=self.cfg.horizon * 2)
        return arrive - s.t

    def _decide(self, s: RobotState, seq):
        cfg = self.cfg
        planned = self._planned(seq, s.t)
        options = []

        # Every option is (kind, cost in seconds, worst predicted gap, route). "Safe" = gap >= safety.
        # Option 1: wait here. Costs the time until resuming the plan is safe.
        hold_gap, _, _ = self._worst(s, ([], {}))
        if hold_gap >= cfg.safety:
            for tau in np.arange(0.0, cfg.horizon, 1.0):
                g, _, _ = self._worst(s, planned, wait=tau)
                if g >= cfg.safety:
                    options.append(("wait", tau, min(hold_gap, g), None))
                    break

        # Option 2: sidestep through a point off to the side of the object's track, then carry on.
        goal = np.asarray(seq.leg.waypoints[-1], float)
        base_time = self._arrival(s, seq.leg.waypoints[seq.wp_index:])
        for tr in self.tracker.active(s.t):
            _, v = tr.estimate()
            for side in _perpendiculars(v, cfg.detour_dirs):
                for dist in cfg.detour_dists:
                    via = s.position + dist * side
                    if not segment_clear(s.position, via, self.planner.obstacles, self.planner.margin):
                        continue
                    try:
                        rest = self.planner.path(via, goal)
                    except Exception:
                        continue
                    route = [via] + rest
                    g, _, _ = self._worst(s, self._planned(seq, s.t, rest=route))
                    extra = self._arrival(s, route) - base_time
                    options.append(("detour", max(extra, 0.0), g, route))

        safe = [o for o in options if o[2] >= cfg.safety]
        if safe:  # cheapest safe option
            kind, cost, gap, route = min(safe, key=lambda o: o[1])
        else:     # nothing keeps the full gap: take the one that keeps the biggest, and reconsider soon
            fallback = ("wait", cfg.rethink, hold_gap, None)  # just hold still
            kind, cost, gap, route = max(options + [fallback], key=lambda o: o[2])
            if kind == "wait":
                cost = cfg.rethink
            self._emit(s.t, "no_safe_option", best_gap_m=round(gap, 3), leg=seq.leg.name)
        self._emit(s.t, "replanned", choice=kind, cost_s=round(float(cost), 1), predicted_gap_m=round(float(gap), 3),
                   options=len(options), safe_options=len(safe), leg=seq.leg.name)
        if kind == "wait":
            self.mode, self._resume_at = "waiting", s.t + cost
        else:
            seq.reroute(route)
            self.mode = "clear"
            self._next_check = s.t + cfg.check_every


def _speed_along(s: RobotState, waypoints) -> float:
    """The robot's current speed in the direction of its first waypoint (0 if moving away)."""
    if not len(waypoints):
        return 0.0
    d = np.asarray(waypoints[0], float) - s.position
    n = np.linalg.norm(d)
    return float(max(0.0, s.velocity @ d / n)) if n > 1e-6 else 0.0


def _perpendiculars(v, n):
    """n unit vectors perpendicular to v (any n directions if v is ~0)."""
    v = np.asarray(v, float)
    if np.linalg.norm(v) < 1e-6:
        v = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    u = np.cross(v, [1.0, 0, 0] if abs(v[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(v, u)
    return [np.cos(a) * u + np.sin(a) * w for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
