"""Thruster pack: turns a commanded wrench into what the hardware can actually do."""

import numpy as np

from ..core.types import VehicleSpec, Wrench

G0 = 9.80665  # m/s^2, standard gravity (only used to convert Isp to exhaust speed)


class ThrusterModel:
    """Idealised cold-gas thruster pack.

    Each body axis can push with at most `max_force` and twist with at most
    `max_torque`. Propellant burn follows the rocket equation's rate form:
        mass flow = thrust / (Isp * g0)
    Torque comes from pairs of thrusters on a moment arm of half the hull
    size, so twisting also costs propellant.

    Simplifications (fine for now, worth revisiting later):
    - every axis can thrust independently and instantly (no valve dynamics),
    - the robot's mass stays constant as propellant burns (3% of mass here),
    - no thruster misalignment or noise.
    """

    def __init__(self, vehicle: VehicleSpec):
        self.vehicle = vehicle
        self.propellant = vehicle.propellant
        self.total_impulse = 0.0  # N*s, all thrust spent (translation + rotation)
        self.linear_impulse = 0.0  # N*s, translation only; divided by mass this is delta-v

    def apply(self, cmd: Wrench, dt: float) -> Wrench:
        v = self.vehicle
        if self.propellant <= 0:
            return Wrench.zero()
        force = np.clip(cmd.force, -v.max_force, v.max_force)
        torque = np.clip(cmd.torque, -v.max_torque, v.max_torque)
        thrust = np.abs(force).sum() + np.abs(torque).sum() / v.half_size
        burn = thrust * dt / (v.isp * G0)
        if burn > self.propellant:  # tank runs dry partway through this step
            scale = self.propellant / burn
            force, torque, thrust, burn = force * scale, torque * scale, thrust * scale, self.propellant
        self.propellant -= burn
        self.total_impulse += thrust * dt
        self.linear_impulse += np.abs(force).sum() * dt
        return Wrench(force, torque)
