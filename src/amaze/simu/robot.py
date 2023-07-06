from dataclasses import dataclass, field
from enum import Enum, auto
from logging import getLogger
from typing import Annotated, Optional, Tuple

import numpy as np

from amaze.simu.controllers.base import BaseController
from amaze.simu.pos import Pos, Vec
from amaze.utils.build_data import BaseBuildData

logger = getLogger(__name__)


class InputType(Enum):
    DISCRETE = auto()
    CONTINUOUS = auto()


class OutputType(Enum):
    DISCRETE = auto()
    CONTINUOUS = auto()


Action = Vec
State = np.ndarray


class Robot:
    @dataclass
    class BuildData(BaseBuildData):
        vision: Annotated[int, "agent vision size"] = 5
        inputs: Annotated[InputType, "Input type"] = InputType.DISCRETE
        outputs: Annotated[OutputType, "Output type"] = OutputType.DISCRETE

        control: Annotated[str, "Control type"] = "random"
        control_data: Optional[dict] = field(default_factory=dict)

        def __post_init__(self):
            if self.control:
                self.control = self.control.upper()

    RADIUS = .1
    INERTIAL_LOSS = .5
    ACCELERATION_SCALE = .5  # RADIUS * 2

    def __init__(self):
        self.pos = None
        self.prev_cell = None

        self.delta_t = None
        self.vel = None
        self.acc = None

        self.reward = None

        self.inputs = None
        self.outputs = None
        self.controller: Optional[BaseController] = None

    def reset(self, pos: Pos):
        assert isinstance(pos, Pos)
        self.pos = pos
        self.prev_cell = pos.aligned()
        self.vel = Vec.null()
        self.acc = Vec.null()
        self.reward = 0

    def set_input_buffer(self, buffer: State):
        self.inputs = buffer

    def set_dt(self, dt):
        self.delta_t = dt

    def cell(self):
        return self.pos.aligned()

    def step(self) -> Action:
        controller_output = self.controller(self.pos, self.inputs)
        if self.delta_t < 1:
            self.acc = self.ACCELERATION_SCALE * controller_output
            # self.acc = Vec(0, 1)
            self.vel = \
                (1 - self.INERTIAL_LOSS * self.delta_t) * self.vel \
                + self.delta_t * self.acc
            if self.vel.length() < min(.01, self.ACCELERATION_SCALE / 2):
                self.outputs = Vec.null()
            else:
                self.outputs = self.delta_t * self.vel

        else:
            self.outputs = controller_output

        return self.outputs

    @staticmethod
    def discrete_actions():
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]
