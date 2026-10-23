from dataclasses import replace

import numpy as np
import pytest

from space_autonomy.core.rotations import look_at_quat, quat_angle, quat_from_axis_angle, quat_mul
from space_autonomy.core.types import VehicleSpec
from space_autonomy.perception.change_detection import assess
from space_autonomy.perception.reference import ReferenceRenderer
from space_autonomy.perception.registration import (ImageAlignment, align_image, backproject,
                                                    refine_camera_pose, warp_view)
from space_autonomy.scenarios.alpha03 import ALPHA03
from space_autonomy.sim.anomalies import inject
from space_autonomy.sim.sensors import OnboardCamera
from space_autonomy.sim.world import SpaceSim

VEHICLE = VehicleSpec()
TARGETS = {t.name: t for t in ALPHA03.targets()}


@pytest.fixture(scope="module")
def reference():
    return ReferenceRenderer(VEHICLE, ALPHA03)


def capture(spacecraft, name, offset=(0, 0, 0)):
    t = TARGETS[name]
    p = t.viewpoint + np.array(offset)
    sim = SpaceSim(VEHICLE, spacecraft, start_pos=p, start_quat=look_at_quat(t.point - p))
    return sim.state(), OnboardCamera(sim, seed=0).capture()


def perturbed(state, pos, att_deg, axis=(0.3, -0.5, 0.8)):
    dq = quat_from_axis_angle(axis, np.radians(att_deg))
    return replace(state, position=state.position + np.array(pos), quat=quat_mul(state.quat, dq))


def register(reference, obs, est):
    cam_R, cam_t = reference.camera_pose(est)
    ref = reference.render(est)
    cam_R, cam_t, info = refine_camera_pose(obs.depth, ref.depth, cam_R, cam_t, reference.fovy)
    refined = reference.state_for_camera(cam_R, cam_t, est) if info.accepted else est
    ref = reference.render(refined)
    return refined, warp_view(ref, align_image(obs, ref))


def test_backprojection_puts_points_on_the_right_surface(reference):
    state, _ = capture(ALPHA03, "star_tracker", offset=(0, 0.4, 0.3))
    ref = reference.render(state)
    cam_R, cam_t = reference.camera_pose(state)
    world = backproject(ref.depth, reference.fovy, 20.0) @ cam_R.T + cam_t
    face = world[np.abs(world[:, 0] - 0.6) < 0.01]  # the bus's +x face is the plane x = 0.6
    assert len(face) > 10000
    assert face[:, 0].std() < 0.001


def test_camera_pose_round_trip(reference):
    state, _ = capture(ALPHA03, "antenna")
    back = reference.state_for_camera(*reference.camera_pose(state), state)
    assert np.allclose(back.position, state.position)
    assert quat_angle(back.quat, state.quat) < 1e-9


def test_icp_recovers_pose_error_on_a_structured_view(reference):
    true, obs = capture(ALPHA03, "star_tracker")
    est = perturbed(true, (0.01, -0.008, 0.005), 0.6)
    refined, _ = register(reference, obs, est)
    assert np.degrees(quat_angle(refined.quat, true.quat)) < 0.1
    assert np.linalg.norm(refined.position - true.position) < 0.005


def test_image_alignment_recovers_a_known_shift(reference):
    state, obs = capture(ALPHA03, "docking_port")
    ref = reference.render(state)
    moved = warp_view(ref, ImageAlignment(True, 3.0, -2.0, 0.3))
    a = align_image(obs, moved)
    # Undoing the shift should land close to zero net motion.
    assert a.accepted
    assert abs(a.dx_px + 3.0) < 0.2 and abs(a.dy_px - 2.0) < 0.2 and abs(a.rotation_deg + 0.3) < 0.05


@pytest.mark.parametrize("name", list(TARGETS))
def test_registration_removes_false_alarms_from_attitude_error(reference, name):
    true, obs = capture(ALPHA03, name)
    est = perturbed(true, (0, 0, 0), 0.5)
    _, ref = register(reference, obs, est)
    assert assess(obs, ref, reference.index(name), name).kind == "nominal"


@pytest.mark.parametrize("kind, name", [("misaligned", "solar_panel_1"), ("missing", "antenna"),
                                        ("foreign_object", "docking_port"), ("surface_damage", "antenna")])
def test_registration_does_not_explain_away_real_faults(reference, kind, name):
    sc, _ = inject(ALPHA03, kind, name, np.random.default_rng(0))
    true, obs = capture(sc, name)
    est = perturbed(true, (0.005, 0.005, -0.005), 0.3)
    _, ref = register(reference, obs, est)
    assert assess(obs, ref, reference.index(name), name).kind == kind

