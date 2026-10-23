import numpy as np

from space_autonomy.core.rotations import quat_angle
from space_autonomy.core.types import RobotState
from space_autonomy.scenarios.conditions import Conditions
from space_autonomy.sim.sensors import NavigationError
from space_autonomy.sim.world import SUN_DIR


def state(t):
    return RobotState(t, np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))


def test_ideal_conditions_change_nothing():
    c = Conditions()
    assert c.navigation(0) is None
    assert np.array_equal(c.sun_dir(0), SUN_DIR)


def test_navigation_error_has_the_requested_spread():
    nav = NavigationError(pos_sigma=0.02, att_sigma=np.radians(0.5), tau=5.0, seed=1)
    errs = [nav.estimate(state(t)) for t in np.arange(0, 5000, 1.0)]
    pos = np.array([e.position for e in errs])
    att = np.array([np.degrees(quat_angle(e.quat, [1, 0, 0, 0])) for e in errs])
    assert np.allclose(pos.std(axis=0), 0.02, rtol=0.15)
    # The angle of a 3D rotation error with 0.5 deg per axis averages 0.5 * sqrt(8 / pi) ~ 0.8 deg.
    assert 0.65 < att.mean() < 0.95


def test_navigation_error_drifts_slowly():
    """Errors 4 s apart should be correlated by exp(-4 / tau) = 0.875 for tau = 30 s."""
    nav = NavigationError(pos_sigma=0.02, tau=30.0, seed=2)
    x = np.array([nav.estimate(state(t)).position[0] for t in np.arange(0, 60000, 4.0)])
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1] - np.exp(-4 / 30)) < 0.03


def test_same_time_gives_same_estimate():
    nav = NavigationError(pos_sigma=0.02, seed=3)
    assert np.array_equal(nav.estimate(state(1.0)).position, nav.estimate(state(1.0)).position)


def test_sun_error_angle():
    for deg in (1, 5, 20):
        d = Conditions(sun_error_deg=deg).sun_dir(seed=4)
        cos = d @ SUN_DIR / (np.linalg.norm(d) * np.linalg.norm(SUN_DIR))
        assert np.isclose(np.degrees(np.arccos(cos)), deg)
