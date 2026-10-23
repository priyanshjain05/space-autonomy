import gc

import numpy as np

from space_autonomy.core.rotations import look_at_quat
from space_autonomy.core.types import VehicleSpec
from space_autonomy.perception.reference import ReferenceRenderer
from space_autonomy.scenarios.alpha03 import ALPHA03
from space_autonomy.sim.world import SpaceSim


def test_dropping_one_renderer_does_not_break_another():
    """Freeing a renderer used to delete the GPU buffers of whichever renderer was current."""
    vehicle = VehicleSpec()
    t = ALPHA03.targets()[0]
    state = SpaceSim(vehicle, ALPHA03, start_pos=t.viewpoint, start_quat=look_at_quat(t.point - t.viewpoint)).state()
    doomed = ReferenceRenderer(vehicle, ALPHA03)
    doomed.render(state)
    survivor = ReferenceRenderer(vehicle, ALPHA03)
    expected = survivor.render(state)  # the survivor's context is now current ...
    del doomed                          # ... while the other renderer is freed
    gc.collect()
    for _ in range(3):
        again = survivor.render(state)
        assert np.array_equal(again.labels, expected.labels)
        assert np.array_equal(again.depth, expected.depth)
