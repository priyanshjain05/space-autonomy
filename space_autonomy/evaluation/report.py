"""Save mission outputs and print a human-readable report."""

import csv
import json
import math
from pathlib import Path


def format_report(summary: dict) -> str:
    s = summary
    yes = lambda b: "YES" if b else "NO"
    lines = [
        "MISSION REPORT",
        "─" * 40,
        f"Mission:               {s['mission']}",
        f"Mission success:       {yes(s['success'])}",
        f"Targets inspected:     {s['targets_inspected']} / {s['targets_expected']}",
        f"Mission time:          {s['mission_time_s']:.0f} s",
        f"Propellant used:       {1000 * s['propellant_used_kg']:.0f} g ({s['propellant_used_pct']}%)",
        f"Delta-v spent:         {s['delta_v_mps']:.2f} m/s",
        f"Distance flown:        {s['path_length_m']:.1f} m",
        f"Collisions:            {s['collisions']}",
        f"Min clearance:         " + (f"{s['min_clearance_m']:.2f} m" if s["min_clearance_m"] is not None else "n/a"),
        f"Human interventions:   {s['human_interventions']}",
    ]
    if "anomalies_injected" in s:
        lines[5:5] = [
            f"Anomalies found:       {s['anomalies_found']} / {s['anomalies_injected']}",
            f"  correct type:        {s['anomaly_type_correct']} / {s['anomalies_found']}",
            f"False alarms:          {s['false_alarms']}",
        ]
    return "\n".join(lines)


def save_run(out_dir, result) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "telemetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.telemetry[0]))
        w.writeheader()
        for r in result.telemetry:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else v) for k, v in r.items()})
    (out / "events.json").write_text(json.dumps(result.events, indent=2))
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2))
    (out / "report.txt").write_text(format_report(result.summary) + "\n")
    return out
