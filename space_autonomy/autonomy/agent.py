"""The inspection agent: flies the route, looks at each target, keeps track of what it found.

    sequencer   where to go and how to point (planning + control)
    perception  line the digital twin up with the camera (registration),
                then compare the two
    world model what the robot believes about each component

The harness talks to the agent through three things only: command(state)
returns thruster commands, `image_request` says "I want a picture now",
and on_image(observation, state) delivers it.
"""

from dataclasses import replace

import numpy as np

from ..core.types import Observation, RobotState, VehicleSpec, Wrench
from ..perception.change_detection import Thresholds, assess
from ..perception.reference import ReferenceRenderer
from ..perception.registration import align_image, refine_camera_pose, warp_view
from ..planning.inspection_planner import fixed_route_inspector
from ..planning.sequencer import LegSequencer
from .world_model import WorldModel


class InspectionAgent:
    def __init__(self, sequencer: LegSequencer, reference: ReferenceRenderer, world_model: WorldModel,
                 thresholds: Thresholds = Thresholds(), registration: bool = True):
        self.sequencer = sequencer
        self.registration = registration
        self.reference = reference
        self.world_model = world_model
        self.thresholds = thresholds
        self.image_request: str | None = None
        self.last = None   # (observation, reference view, assessment) of the latest look, for debugging
        self._events = []

    @property
    def done(self) -> bool:
        return self.sequencer.done

    @property
    def legs(self):
        return self.sequencer.legs

    def status(self) -> dict:
        return self.sequencer.status()

    def pop_events(self) -> list:
        ev, self._events = self._events, []
        return ev

    def command(self, state: RobotState) -> Wrench:
        wrench = self.sequencer.command(state)
        for ev in self.sequencer.pop_events():
            self._events.append(ev)
            if ev["event"] == "arrived" and self.sequencer.leg.kind == "inspect":
                self.image_request = ev["leg"]
        return wrench

    def on_image(self, obs: Observation, state: RobotState):
        name = self.image_request
        self.image_request = None
        ref = self.reference.render(state)
        event = {"t": round(state.t, 3), "event": "assessed", "leg": name,
                 "est_position": state.position.tolist(), "est_quat": state.quat.tolist()}
        if self.registration:
            ref, refined, reg, img = self._register(obs, ref, state)
            event.update(refined_position=refined.position.tolist(), refined_quat=refined.quat.tolist(),
                         icp_accepted=reg.accepted, icp_shift_m=round(reg.shift_m, 4),
                         icp_rotation_deg=round(reg.rotation_deg, 3), image_accepted=img.accepted,
                         image_shift_px=round(float(np.hypot(img.dx_px, img.dy_px)), 2),
                         image_rotation_deg=round(img.rotation_deg, 3))
        a = assess(obs, ref, self.reference.index(name), name, self.thresholds)
        self.world_model.update(a)
        self.last = (obs, ref, a)
        event.update(result=a.kind, confidence=round(a.confidence, 2))
        self._events.append(event)

    def _register(self, obs, ref, state):
        """Two-stage alignment of the twin to the camera: 3D (depth ICP), then 2D (image outline)."""
        cam_R, cam_t = self.reference.camera_pose(state)
        raw = ref
        cam_R, cam_t, reg = refine_camera_pose(obs.depth, ref.depth, cam_R, cam_t, self.reference.fovy)
        refined = state
        if reg.accepted:
            refined = self.reference.state_for_camera(cam_R, cam_t, state)
            ref = self.reference.render(refined)
        img = align_image(obs, ref)
        if reg.accepted and not img.accepted:
            # The stages disagree: after the 3D correction the picture still needs an implausibly
            # big shift. The 3D fit was probably pulled by a fault in view, so don't trust it.
            reg = replace(reg, accepted=False)
            refined, ref = state, raw
            img = align_image(obs, ref)
        return warp_view(ref, img), refined, reg, img


def make_inspection_agent(vehicle: VehicleSpec, nominal, home, order, registration=True) -> InspectionAgent:
    """Baseline 1 + perception: fixed route, model-based change detection at every stop."""
    seq = fixed_route_inspector(vehicle, nominal.targets(), nominal.obstacles(), home, order=order)
    return InspectionAgent(seq, ReferenceRenderer(vehicle, nominal), WorldModel(order), registration=registration)
