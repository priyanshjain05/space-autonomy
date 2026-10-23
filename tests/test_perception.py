import numpy as np
import pytest

from space_autonomy.core.rotations import look_at_quat
from space_autonomy.core.types import VehicleSpec
from space_autonomy.perception.change_detection import assess
from space_autonomy.perception.reference import ReferenceRenderer
from space_autonomy.scenarios.alpha03 import ALPHA03
from space_autonomy.sim.anomalies import inject
from space_autonomy.sim.sensors import OnboardCamera
from space_autonomy.sim.world import SpaceSim

VEHICLE = VehicleSpec()
TARGETS = {t.name: t for t in ALPHA03.targets()}


@pytest.fixture(scope="module")
def reference():
    return ReferenceRenderer(VEHICLE, ALPHA03)


def look(reference, spacecraft, name):
    """Park the robot at `name`'s viewpoint in `spacecraft`, take a picture, assess it."""
    t = TARGETS[name]
    sim = SpaceSim(VEHICLE, spacecraft, start_pos=t.viewpoint, start_quat=look_at_quat(t.point - t.viewpoint))
    obs = OnboardCamera(sim, seed=0).capture()
    return assess(obs, reference.render(sim.state()), reference.index(name), name)


@pytest.mark.parametrize("name", list(TARGETS))
def test_healthy_spacecraft_raises_no_alarm(reference, name):
    assert look(reference, ALPHA03, name).kind == "nominal"


@pytest.mark.parametrize("kind, name", [
    ("missing", "antenna"),
    ("misaligned", "solar_panel_1"),
    ("foreign_object", "thruster_2"),
    ("surface_damage", "docking_port"),
])
def test_each_anomaly_type_is_recognised(reference, kind, name):
    sc, _ = inject(ALPHA03, kind, name, np.random.default_rng(0))
    a = look(reference, sc, name)
    assert a.kind == kind
    assert a.confidence > 0.9


def test_neighbours_anomaly_is_not_blamed_on_the_component(reference):
    """A tilted solar panel pokes into the antenna's view; the antenna itself is fine."""
    rng = np.random.default_rng(1003)
    sc, _ = inject(ALPHA03, "misaligned", "solar_panel_2", rng)
    for _ in range(20):  # a few tilts, some of which reach into the antenna's view
        assert look(reference, sc, "antenna").kind == "nominal"
        sc, _ = inject(ALPHA03, "misaligned", "solar_panel_2", rng)
