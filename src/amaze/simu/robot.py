from dataclasses import dataclass, field
from logging import getLogger
from typing import Annotated, Optional, Tuple

from amaze.simu.types import InputType, OutputType
from amaze.simu.pos import Pos, Vec
from amaze.simu._build_data import BaseBuildData

logger = getLogger(__name__)


class Robot:
    """ The virtual robot """
    @dataclass
    class BuildData(BaseBuildData):
        """ Structure describing the agent's parameters
        """
        inputs: Annotated[InputType, "Input type"] = InputType.DISCRETE
        outputs: Annotated[OutputType, "Output type"] = OutputType.DISCRETE
        vision: Annotated[int, "agent vision size"] = 15

        control: Annotated[str, "Controller type"] = "random"
        control_data: Optional[dict] = field(default_factory=dict)

        __inputs_to_string = {i: i.name[0] for i in InputType}
        __outputs_to_string = {o: o.name[0] for o in OutputType}

        __string_to_input = {v: k for k, v in __inputs_to_string.items()}
        __string_to_output = {v: k for k, v in __outputs_to_string.items()}

        def __post_init__(self):
            if not isinstance(self.control, BaseBuildData.Unset):
                self.control = self.control.lower()
                # assert self.control in CONTROLLERS.keys(), self.control

        @classmethod
        def from_string(cls, robot: str):
            """ Parses a string into input/output types and, optionally, vision
            size

            Format is: IO[V]

            where the input character I is taken from :class:`.InputType` and
            be either D (:const:`~amaze.simu.types.InputType.DISCRETE`)
            or C (:const:`~amaze.simu.types.InputType.CONTINUOUS`).
            Similarly, the output character O is taken from :class:`.OutputType`
            and can also either be C or D.
            In case of continuous inputs, V provides the size of agent's retina
            as an *odd* integer

            :raises ValueError: if requesting DC mode or if the retina size is
             even
            """
            bd = cls()
            assert len(robot) >= 2
            bd.inputs = cls.__string_to_input[robot[0]]
            bd.outputs = cls.__string_to_output[robot[1]]
            if (bd.inputs == InputType.DISCRETE
                    and bd.outputs == OutputType.CONTINUOUS):
                raise ValueError("Incompatible hybrid mode. Agent cannot have"
                                 "discrete inputs and continuous outputs")

            if bd.inputs is InputType.CONTINUOUS and len(robot) > 2:
                bd.vision = int(robot[2:])
                if (bd.vision % 2) != 1:
                    raise ValueError("Retina size must be odd")
            return bd

        def to_string(self):
            """ Generates a string for the input/output types and, if relevant,
            vision size"""
            s = (self.__inputs_to_string[self.inputs]
                 + self.__outputs_to_string[self.outputs])
            if self.inputs is InputType.CONTINUOUS:
                s += str(self.vision)
            return s

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
