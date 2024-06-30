from dataclasses import dataclass
from logging import getLogger
from typing import Annotated, Optional, Tuple

from ._build_data import BaseBuildData
from .pos import Pos, Vec
from .types import InputType, OutputType

logger = getLogger(__name__)


class Robot:
    """The virtual robot"""

    @dataclass
    class BuildData(BaseBuildData):
        """Structure describing the agent's parameters"""

        inputs: Annotated[InputType, "Input type"] = InputType.DISCRETE
        outputs: Annotated[OutputType, "Output type"] = OutputType.DISCRETE
        vision: Annotated[Optional[int], "agent vision size"] = 15

        __inputs_to_string = {i: i.name[0] for i in InputType}
        __outputs_to_string = {o: o.name[0] for o in OutputType}

        __string_to_input = {v: k for k, v in __inputs_to_string.items()}
        __string_to_output = {v: k for k, v in __outputs_to_string.items()}

        def __post_init__(self):
            self._post_init(allow_unset=False)

        def _post_init(self, allow_unset: bool):
            def assert_ok(*args):
                self._assert_field_type(*args, allow_unset=allow_unset)

            assert_ok("inputs")
            assert_ok("outputs")

            if self.vision is not None:
                assert_ok("vision")

            if self.inputs is not InputType.CONTINUOUS:
                self.vision = None
            elif self.vision is None:
                self.vision = self.__class__.vision

        @classmethod
        def from_string(cls, robot: str, overrides: Optional["Robot.BuildData"] = None):
            """Parses a string into input/output types and, optionally, vision
            size

            Format is: IO[V] or S[V]

            where the input character I is taken from :class:`.InputType` and
            be either D (:const:`~amaze.simu.types.InputType.DISCRETE`)
            or C (:const:`~amaze.simu.types.InputType.CONTINUOUS`).
            Similarly, the output character O is taken from
            :class:`.OutputType` and can also either be C or D.
            Shorthands (S) are also available with D, H, and C corresponding to
            DD, CD, and CC, respectively.
            In case of continuous inputs, V provides the size of agent's retina
            as an *odd* integer

            :raises TypeError: if passing invalid parameters
            :raises ValueError: if requesting DC mode or if the retina size is
             even
            """
            bd = cls()

            if robot[0] not in ["D", "C", "H"]:
                raise TypeError(f"Invalid token[0]: {robot[0]} should be 'D'," f" 'C' or 'H'")

            if len(robot) > 1 and robot[1].isalpha() and robot[1] not in ["D", "C"]:
                raise TypeError(f"Invalid token[1]: {robot[1]} should be 'D'" f" or 'C'")

            ix = 1 + (len(robot) > 1 and robot[1].isalpha())
            if any(not c.isdigit() for c in robot[ix:]):
                raise TypeError(f"Invalid token[ix:]: {robot[ix:]} should be" f" a digit")

            io = robot[0:ix]
            if len(io) == 1:
                bd.inputs, bd.outputs = {
                    "D": (InputType.DISCRETE, OutputType.DISCRETE),
                    "H": (InputType.CONTINUOUS, OutputType.DISCRETE),
                    "C": (InputType.CONTINUOUS, OutputType.CONTINUOUS),
                }[io]
            else:
                bd.inputs = cls.__string_to_input[io[0]]
                bd.outputs = cls.__string_to_output[io[1]]
                if bd.inputs == InputType.DISCRETE and bd.outputs == OutputType.CONTINUOUS:
                    raise ValueError(
                        "Incompatible hybrid mode. Agent cannot"
                        " have discrete inputs and continuous"
                        " outputs"
                    )

            if bd.inputs is InputType.CONTINUOUS:
                if ix < len(robot) > 2:
                    bd.vision = int(robot[ix:])
                    if (bd.vision % 2) != 1:
                        raise ValueError("Retina size must be odd")
                else:
                    bd.vision = cls.vision

            if overrides:
                return bd.override_with(overrides)
            else:
                return bd

        @classmethod
        def from_controller(cls, controller: "BaseController") -> "Robot.BuildData":  # noqa: F821
            """Create a robot build data from an existing controller"""

            bd = cls()
            bd.inputs = controller.input_type
            bd.outputs = controller.output_type

            if bd.inputs is InputType.CONTINUOUS:
                if (v := controller.vision) is not None:
                    bd.vision = v
                else:  # pragma no cover
                    raise ValueError("Controller does not provide vision size")

            return bd

        def to_string(self):
            """Generates a string for the input/output types and, if relevant,
            vision size"""
            s = self.__inputs_to_string[self.inputs] + self.__outputs_to_string[self.outputs]
            if self.inputs is InputType.CONTINUOUS:
                s += str(self.vision)
            return s

    RADIUS = 0.1
    INERTIAL_LOSS = 0.5
    ACCELERATION_SCALE = 0.5  # RADIUS * 2

    def __init__(self, data: BuildData):
        self.data = data

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
        self.vel = (1 - self.INERTIAL_LOSS * dt) * self.vel + dt * self.acc

        if self.vel.length() < min(0.01, self.ACCELERATION_SCALE / 2):
            self.vel = Vec.null()
        return self.pos + dt * self.vel

    def to_dict(self):
        return dict(pos=self.pos, vel=self.vel, acc=self.acc)
