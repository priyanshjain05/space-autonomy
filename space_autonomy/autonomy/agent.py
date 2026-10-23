"""The inspection agent: flies the route, looks at each target, keeps track of what it found.

    sequencer   where to go and how to point (planning + control)
    perception  line the digital twin up with the camera (registration),
                then compare the two
    world model what the robot believes about each component
    active      when a look is inconclusive, go and look again from elsewhere (M7)
    hazards     when loose debris is about to get in the way: stop, look, replan (M8/M9)

The harness talks to the agent through three things only: command(state)
returns thruster commands, `image_request` says "I want a picture now",
and on_image(observation, state) delivers it.
"""

from dataclasses import dataclass, replace

import numpy as np

from ..controllers.pose_controller import PoseController
from ..core.types import Observation, RobotState, VehicleSpec, Wrench
from ..perception.calibration import ConfidenceModel, features
from ..perception.change_detection import UNOBSERVABLE, Thresholds, assess
from ..perception.reference import ReferenceRenderer
from ..perception.registration import align_image, refine_camera_pose, warp_view
from ..planning.inspection_planner import InspectionPlanner, vehicle_margin
from ..planning.sequencer import LegSequencer
from .hazards import HazardConfig, HazardResponder
from .world_model import WorldModel


@dataclass(frozen=True)
class ActiveInspection:
    """Look again when unsure (M7).

    After each look the robot keeps its most confident verdict so far. If that
    is still below `threshold`, it flies to another viewpoint (the usual view
    direction tilted by `tilt_deg`) and looks again, up to `max_extra_looks`
    times, before moving on. This only makes sense with a calibrated
    confidence: the old heuristic was ~100% sure of almost everything.
    """
    threshold: float = 0.9
    max_extra_looks: int = 2
    tilt_deg: float = 30.0
    dwell: float = 2.0   # s held at a re-inspection viewpoint


class InspectionAgent:
    def __init__(self, sequencer: LegSequencer, reference: ReferenceRenderer, world_model: WorldModel,
                 thresholds: Thresholds = Thresholds(), registration: bool = True,
                 confidence_model: ConfidenceModel | None = None, active: ActiveInspection | None = None,
                 planner: InspectionPlanner | None = None, targets=(), hazards: HazardResponder | None = None):
        self.sequencer = sequencer
        self.hazards = hazards        # None: ignore loose objects entirely
        self.active = active          # None: one look per component, as planned
        self.planner = planner        # needed for active inspection: plans the detours
        self.targets = {t.name: t for t in targets}
        self.looks = {}               # component -> every assessment so far
        self._viewpoints = {}         # component -> viewpoints already used
        self.registration = registration
        self.confidence_model = confidence_model  # None: keep the heuristic confidence
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

    def on_detections(self, detections):
        if self.hazards is not None:
            self.hazards.on_detections(detections)

    def command(self, state: RobotState) -> Wrench:
        if self.hazards is not None:
            hold = self.hazards.step(state, self.sequencer)
            self._events += self.hazards.events
            self.hazards.events.clear()
            if hold is not None:  # the hazard responder has the controls: hold position
                return self.sequencer.controller.command(state, *hold)
        wrench = self.sequencer.command(state)
        for ev in self.sequencer.pop_events():
            self._events.append(ev)
            leg = self.sequencer.leg
            if ev["event"] == "arrived" and leg.kind in ("inspect", "reinspect") and not leg.assessed:
                leg.assessed = True
                self.image_request = ev["leg"]
        return wrench

    def on_image(self, obs: Observation, state: RobotState):
        name = self.image_request
        self.image_request = None
        ref = self.reference.render(state)
        event = {"t": round(state.t, 3), "event": "assessed", "leg": name,
                 "est_position": state.position.tolist(), "est_quat": state.quat.tolist()}
        reg = img = None
        fallback = False
        if self.registration:
            ref, refined, reg, img, fallback = self._register(obs, ref, state)
            event.update(refined_position=refined.position.tolist(), refined_quat=refined.quat.tolist(),
                         icp_accepted=reg.accepted, icp_shift_m=round(reg.shift_m, 4),
                         icp_rotation_deg=round(reg.rotation_deg, 3), image_accepted=img.accepted,
                         image_shift_px=round(float(np.hypot(img.dx_px, img.dy_px)), 2),
                         image_rotation_deg=round(img.rotation_deg, 3))
        a = assess(obs, ref, self.reference.index(name), name, self.thresholds)
        f = features(a, self.thresholds.min_px, reg, img, fallback)
        event.update(result=a.kind, heuristic_confidence=round(a.confidence, 4), features=f)
        if self.confidence_model is not None and a.kind != UNOBSERVABLE:
            a = replace(a, confidence=self.confidence_model.predict(f, a.kind))
        self.last = (obs, ref, a)

        # Fuse: of all the looks at this component, believe the most confident one. With a calibrated
        # confidence, "most confident" means "most likely right", so this is a fair comparison.
        looks = self.looks.setdefault(name, [])
        looks.append(a)
        best = max(looks, key=lambda x: x.confidence)
        self.world_model.update(best, observations=len(looks))
        event.update(confidence=round(a.confidence, 4), look=len(looks), believed=best.kind,
                     believed_confidence=round(best.confidence, 4))
        self._events.append(event)

        act = self.active
        if act is not None and best.confidence < act.threshold and len(looks) <= act.max_extra_looks:
            self._look_again(name, state.t)

    def _look_again(self, name, t):
        """Insert a detour to another viewpoint of `name`, right after the current stop."""
        target = self.targets[name]
        here = np.asarray(self.sequencer.leg.waypoints[-1], float)
        used = self._viewpoints.setdefault(name, [target.viewpoint])
        vp = self.planner.nearest_viewpoint(target, here, tilt_deg=self.active.tilt_deg, avoid=used)
        if vp is None:
            self._events.append({"t": round(t, 3), "event": "reinspect_unavailable", "leg": name})
            return
        used.append(vp)
        leg = self.planner.inspect_leg(target, vp, here, dwell=self.active.dwell, kind="reinspect")
        self.sequencer.detour(leg, lambda old, start: replace(old, waypoints=self.planner.path(start, old.waypoints[-1])))
        self._events.append({"t": round(t, 3), "event": "reinspect_planned", "leg": name,
                             "viewpoint": np.round(vp, 3).tolist()})

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
        fallback = reg.accepted and not img.accepted
        if fallback:
            # The stages disagree: after the 3D correction the picture still needs an implausibly
            # big shift. The 3D fit was probably pulled by a fault in view, so don't trust it.
            reg = replace(reg, accepted=False)
            refined, ref = state, raw
            img = align_image(obs, ref)
        return warp_view(ref, img), refined, reg, img, fallback


def make_inspection_agent(vehicle: VehicleSpec, nominal, home, order, registration=True,
                          calibrated=True, active=True, avoid=True,
                          hazard_config: HazardConfig = HazardConfig()) -> InspectionAgent:
    """Fixed route + model-based change detection at every stop.

    calibrated  use the learned confidence model (perception/confidence_model.json) if it exists.
                It was trained on the registered pipeline, so it's only used with registration on.
    active      look again from another viewpoint when unsure (needs the calibrated model).
    avoid       react to loose debris reported by the hazard sensor.
    """
    planner = InspectionPlanner(nominal.obstacles(), vehicle_margin(vehicle))
    seq = LegSequencer(planner.fixed_route(nominal.targets(), home, order), PoseController(vehicle))
    model = ConfidenceModel.load() if calibrated and registration else None
    return InspectionAgent(seq, ReferenceRenderer(vehicle, nominal), WorldModel(order),
                           registration=registration, confidence_model=model,
                           active=ActiveInspection() if active and model is not None else None,
                           planner=planner, targets=nominal.targets(),
                           hazards=HazardResponder(planner, vehicle.bounding_radius, hazard_config) if avoid else None)
