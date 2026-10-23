"""Data passed between the simulator and the autonomy stack.

These types are the only contract between the two sides. The simulator
produces a RobotState and accepts a Wrench; the autonomy never touches
MuJoCo directly. Swapping MuJoCo for Isaac Sim later means re-implementing
the simulator side only.
"""

from dataclasses import dataclass

import numpy as np

from .rotations import quat_to_mat


@dataclass(frozen=True)
class RobotState:
    t: float                       # s
    position: np.ndarray           # world frame, m
    quat: np.ndarray               # body -> world, (w, x, y, z)
    velocity: np.ndarray           # world frame, m/s
    angular_velocity: np.ndarray   # body frame, rad/s

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_mat(self.quat)


@dataclass(frozen=True)
class Wrench:
    """Force and torque the thrusters should produce, in the robot's body frame."""
    force: np.ndarray    # N
    torque: np.ndarray   # N*m

    @staticmethod
    def zero() -> "Wrench":
        return Wrench(np.zeros(3), np.zeros(3))


@dataclass(frozen=True)
class VehicleSpec:
    """Datasheet for the free-flying inspector.

    Defaults are in the range of NASA's Astrobee (a ~10 kg cube-shaped
    free-flyer on the ISS) but with a cold-gas thruster pack instead of fans.
    """
    mass: float = 10.0            # kg
    half_size: float = 0.15       # m, the hull is a 0.3 m cube
    max_force: float = 1.0        # N per body axis
    max_torque: float = 0.1       # N*m per body axis
    isp: float = 60.0             # s, typical for cold-gas (nitrogen) thrusters
    propellant: float = 0.3       # kg at launch

    @property
    def inertia(self) -> np.ndarray:
        """Principal moments of a solid cube: m * (2h)^2 / 6 about each axis."""
        return np.full(3, self.mass * (2 * self.half_size) ** 2 / 6)

    @property
    def bounding_radius(self) -> float:
        return float(np.sqrt(3) * self.half_size)
