"""The inspection agent: flies the route, looks at each target, keeps track of what it found.

    sequencer   where to go and how to point (planning + control)
    perception  compare the onboard image with the digital twin
    world model what the robot believes about each component

The harness talks to the agent through three things only: command(state)
returns thruster commands, `image_request` says "I want a picture now",
and on_image(observation, state) delivers it.
"""

from ..core.types import Observation, RobotState, VehicleSpec, Wrench
from ..perception.change_detection import Thresholds, assess
from ..perception.reference import ReferenceRenderer
from ..planning.inspection_planner import fixed_route_inspector
from ..planning.sequencer import LegSequencer
from .world_model import WorldModel


class InspectionAgent:
    def __init__(self, sequencer: LegSequencer, reference: ReferenceRenderer, world_model: WorldModel,
                 thresholds: Thresholds = Thresholds()):
        self.sequencer = sequencer
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
        a = assess(obs, ref, self.reference.index(name), name, self.thresholds)
        self.world_model.update(a)
        self.last = (obs, ref, a)
        self._events.append({"t": round(state.t, 3), "event": "assessed", "leg": name,
                             "result": a.kind, "confidence": round(a.confidence, 2)})


def make_inspection_agent(vehicle: VehicleSpec, nominal, home, order) -> InspectionAgent:
    """Baseline 1 + perception: fixed route, model-based change detection at every stop."""
    seq = fixed_route_inspector(vehicle, nominal.targets(), nominal.obstacles(), home, order=order)
    return InspectionAgent(seq, ReferenceRenderer(vehicle, nominal), WorldModel(order))
