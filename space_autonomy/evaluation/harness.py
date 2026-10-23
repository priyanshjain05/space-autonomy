"""Mission harness: runs an autonomy stack inside the simulator and records what happened.

This loop is the heart of the project:

    sim.state() -> autonomy.command(state) -> sim.step(wrench) -> log -> repeat

The harness is the only place that sees both sides. It measures things the
autonomy itself can't be trusted to report: true collisions, true
clearance, propellant actually burned.
"""

from dataclasses import dataclass

import numpy as np

from ..core.rotations import quat_angle


@dataclass
class MissionResult:
    summary: dict
    telemetry: list   # one dict per log tick
    events: list


def run_mission(sim, autonomy, *, max_time, expected_targets=(), control_hz=20.0, log_hz=10.0,
                on_event=None, on_tick=None, name="mission") -> MissionResult:
    """
    on_event(event, sim)  called for every autonomy event (e.g. to grab a camera image)
    on_tick(sim, rows)    called every log tick (e.g. to record video or drive a viewer)
    """
    dt = 1.0 / control_hz
    rows, events = [], []
    collisions, touching = 0, False
    min_clear = float("inf")
    next_log = 0.0
    start_pos = sim.state().position

    while sim.t < max_time and not autonomy.done:
        state = sim.state()
        cmd = autonomy.command(state)
        touched = sim.step(cmd, dt)
        if touched and not touching:
            collisions += 1   # count each new contact episode once
        touching = touched

        for ev in autonomy.pop_events():
            events.append(ev)
            if on_event:
                on_event(ev, sim)

        if sim.t + 1e-9 >= next_log:
            next_log += 1.0 / log_hz
            clear = sim.clearance()
            min_clear = min(min_clear, clear)
            rows.append(_row(sim, autonomy.status(), clear))
            if on_tick:
                on_tick(sim, rows)

    inspected = [e["leg"] for e in events if e["event"] == "inspected"]
    positions = np.array([[r["x"], r["y"], r["z"]] for r in rows])
    v = sim.vehicle
    used = v.propellant - sim.thrusters.propellant
    end_pos = sim.state().position
    summary = {
        "mission": name,
        "success": bool(autonomy.done and collisions == 0 and set(expected_targets) <= set(inspected)),
        "complete": autonomy.done,
        "targets_expected": len(expected_targets),
        "targets_inspected": len(set(inspected) & set(expected_targets)) if expected_targets else len(inspected),
        "mission_time_s": round(sim.t, 2),
        "propellant_used_kg": round(used, 4),
        "propellant_used_pct": round(100 * used / v.propellant, 1),
        "delta_v_mps": round(sim.thrusters.linear_impulse / v.mass, 3),
        "path_length_m": round(float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()), 2),
        "collisions": collisions,
        "min_clearance_m": round(min_clear, 3) if np.isfinite(min_clear) else None,
        "final_distance_from_start_m": round(float(np.linalg.norm(end_pos - start_pos)), 3),
        "human_interventions": 0,
    }
    return MissionResult(summary, rows, events)


def _row(sim, status, clear):
    s = sim.state()
    w = sim.last_applied
    row = {"t": round(s.t, 3)}
    row.update(zip("xyz", s.position))
    row.update(zip(("qw", "qx", "qy", "qz"), s.quat))
    row.update(zip(("vx", "vy", "vz"), s.velocity))
    row.update(zip(("wx", "wy", "wz"), s.angular_velocity))
    row.update(zip(("fx", "fy", "fz"), w.force))
    row.update(zip(("tx", "ty", "tz"), w.torque))
    row["propellant_kg"] = sim.thrusters.propellant
    row["clearance_m"] = clear
    row["leg"] = status["leg"]
    row["phase"] = status["phase"]
    tp, tq = status["target_pos"], status["target_quat"]
    row["pos_err_m"] = float(np.linalg.norm(tp - s.position)) if tp is not None else np.nan
    row["att_err_deg"] = float(np.degrees(quat_angle(s.quat, tq))) if tq is not None else np.nan
    return row
