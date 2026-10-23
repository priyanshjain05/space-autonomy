"""Alpha-03: a deliberately simple target spacecraft, and the nominal scenario around it.

Spacecraft frame: origin at the bus centre, +z "up" (antenna side),
solar panel wings along +/-y, docking port on -x, thrusters on -z.
"""

from dataclasses import dataclass

import numpy as np

from ..core.types import VehicleSpec
from ..sim.spacecraft import Component, Spacecraft

GOLD = (0.85, 0.65, 0.25, 1)      # multi-layer insulation blanket
PANEL = (0.12, 0.2, 0.55, 1)
GREY = (0.6, 0.6, 0.62, 1)
WHITE = (0.92, 0.92, 0.92, 1)
DARK = (0.12, 0.12, 0.12, 1)

ALPHA03 = Spacecraft("Alpha-03", (
    Component("bus", "box", (0, 0, 0), (0.6, 0.6, 0.8), GOLD),
    Component("yoke_1", "box", (0, 0.65, 0), (0.04, 0.05, 0.04), GREY),
    Component("yoke_2", "box", (0, -0.65, 0), (0.04, 0.05, 0.04), GREY),
    Component("solar_panel_1", "box", (0, 2.1, 0), (0.5, 1.4, 0.02), PANEL, inspect_from=(0, 0, 1), standoff=1.6),
    Component("solar_panel_2", "box", (0, -2.1, 0), (0.5, 1.4, 0.02), PANEL, inspect_from=(0, 0, 1), standoff=1.6),
    Component("antenna_mast", "cylinder", (0, 0, 0.95), (0.03, 0.15), GREY),
    Component("antenna", "cylinder", (0, 0, 1.13), (0.4, 0.03), WHITE, inspect_from=(0.5, 0, 1), standoff=1.6),
    Component("star_tracker", "box", (0.65, 0.3, 0.45), (0.05, 0.08, 0.08), DARK, inspect_from=(1, 0, 0), standoff=1.3),
    Component("docking_port", "cylinder", (-0.7, 0, 0), (0.25, 0.1), GREY, axis=(1, 0, 0),
              inspect_from=(-1, 0, 0), standoff=1.5),
    Component("thruster_1", "cylinder", (0.3, 0, -0.9), (0.08, 0.1), DARK, inspect_from=(0, 0, -1), standoff=1.5),
    Component("thruster_2", "cylinder", (-0.3, 0, -0.9), (0.08, 0.1), DARK, inspect_from=(0, 0, -1), standoff=1.5),
))


@dataclass(frozen=True)
class Scenario:
    name: str
    spacecraft: Spacecraft
    vehicle: VehicleSpec
    home: np.ndarray
    route: tuple        # visiting order for the fixed-route baseline
    max_time: float     # s, mission is a failure if not complete by then


def nominal() -> Scenario:
    return Scenario(
        name="alpha03-nominal",
        spacecraft=ALPHA03,
        vehicle=VehicleSpec(),
        home=np.array([-6.0, -3.0, 1.5]),
        route=("star_tracker", "solar_panel_1", "antenna", "solar_panel_2",
               "docking_port", "thruster_2", "thruster_1"),
        max_time=1200.0,
    )
