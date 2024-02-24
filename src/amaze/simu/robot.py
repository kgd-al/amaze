from dataclasses import dataclass, field
from logging import getLogger
from typing import Annotated, Optional, Tuple

from amaze.simu.controllers.control import CONTROLLERS
from amaze.simu.types import InputType, OutputType
from amaze.simu.pos import Pos, Vec
from amaze.utils.build_data import BaseBuildData

logger = getLogger(__name__)


class Robot:
    """ The virtual robot """
    @dataclass
    class BuildData(BaseBuildData):
        """ Structure describing the agent's parameters
        """
        vision: Annotated[int, "agent vision size"] = 15
        inputs: Annotated[InputType, "Input type"] = InputType.DISCRETE
        outputs: Annotated[OutputType, "Output type"] = OutputType.DISCRETE

        control: Annotated[str, "Controller type"] = "random"
        control_data: Optional[dict] = field(default_factory=dict)

        def __post_init__(self):
            if not isinstance(self.control, BaseBuildData.Unset):
                self.control = self.control.lower()
                # assert self.control in CONTROLLERS.keys(), self.control

    RADIUS = .1
    INERTIAL_LOSS = .5
    ACCELERATION_SCALE = .5  # RADIUS * 2

    def __init__(self):
        self.pos = None
        self.prev_cell = None

        self.vel = None
        self.acc = None

        self.reward = None

    def reset(self, pos: Pos):
        assert isinstance(pos, Pos)
        self.pos = pos
        self.prev_cell = pos.aligned()
        self.vel = Vec.null()
        self.acc = Vec.null()
        self.reward = 0

    def cell(self) -> Tuple[int, int]:
        return self.pos.aligned()

    def next_position(self, action, dt) -> Pos:
        self.acc = self.ACCELERATION_SCALE * action
        # self.acc = Vec(0, 1)
        self.vel = \
            (1 - self.INERTIAL_LOSS * dt) * self.vel + dt * self.acc

        if self.vel.length() < min(.01, self.ACCELERATION_SCALE / 2):
            self.vel = Vec.null()
        return self.pos + dt * self.vel
