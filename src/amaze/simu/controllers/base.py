from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from zipfile import ZipFile

from ..pos import Vec
from ..robot import Robot
from ..types import InputType, OutputType, State, Action, classproperty


class BaseController(ABC):
    _simple = True
    _cheats = False
    _savable = True

    infos: dict = {}
    """ Generic storage for additional information.

    For instance the class of mazes this agent should solve.
    """

    _discrete_actions = [
        Action(1, 0),
        Action(0, 1),
        Action(-1, 0),
        Action(0, -1),
    ]

    def __init__(self, robot_data: Robot.BuildData):
        self._robot_data = robot_data

        def test(name, field_type, field_types):
            if field_type not in field_types:
                raise ValueError(
                    f"{name} type not supported."
                    f" {field_type} not in {field_types}: either"
                    f" you passed a wrong argument or you wrongly"
                    f" specified the valid {name.lower()} types"
                    f" for this controller"
                )

        test("Input", self.input_type, self.inputs_types())
        test("Output", self.output_type, self.outputs_types())

    @classproperty
    def is_simple(cls) -> bool:
        """Whether this controller has no internal state (table, ANN, ...)
        or solely used for demonstration and testing"""
        return cls._simple

    @classproperty
    def cheats(cls) -> bool:
        """Whether this controller uses simulation-level information.

        Only meant for test controller *not* for trained agents

        .. seealso:: `~amaze.simu.controllers.CheaterController`
        """
        return cls._cheats

    @classproperty
    def savable(cls) -> bool:
        """Whether this controller contains a savable state.

        If True, implies that `save_to_archive` and `load_from_archive` are
        implemented
        """
        return cls._savable

    @property
    def input_type(self) -> InputType:
        return self._robot_data.inputs

    @property
    def output_type(self) -> OutputType:
        return self._robot_data.outputs

    @property
    def vision(self) -> int:
        return self._robot_data.vision

    @property
    def robot_data(self) -> Robot.BuildData:
        return Robot.BuildData(inputs=self.input_type, outputs=self.output_type, vision=self.vision)

    @abstractmethod
    def __call__(self, inputs: State) -> Vec:
        """
        Generate next move

        :param inputs: Current local information

        :return: (dx,dy) in [-1,1]x[-1,1], the requested movement
        """

    # noinspection PyMethodMayBeStatic
    def details(self) -> dict:
        return {}

    @classproperty
    def name(cls) -> str:
        return cls.__name__

    @classproperty
    def short_name(cls) -> str:
        return cls.name.replace("Controller", "").lower()

    @abstractmethod
    def reset(self):
        pass  # pragma no cover

    @staticmethod
    @abstractmethod
    def inputs_types() -> List[InputType]:  # pragma: no cover
        """Specify what kind of inputs this controller can handle.

        Abstract method that should be implemented and documented.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def outputs_types() -> List[OutputType]:  # pragma: no cover
        """Specify what kind of outputs this controller can handle.

        Abstract method that should be implemented and documented.
        """
        raise NotImplementedError

    @classproperty
    def discrete_actions(cls) -> List[Action]:
        """Specifies the set of discrete actions that can be taken by an agent"""
        return cls._discrete_actions

    def _save_to_archive(self, archive: ZipFile, *args, **kwargs) -> bool:  # pragma: no cover
        """Re-implement to save derived-specific content to the archive

        :meta public:
        """
        raise NotImplementedError

    @classmethod
    def _load_from_archive(
        cls, archive: ZipFile, robot: Robot.BuildData, *args, **kwargs
    ) -> "BaseController":  # pragma: no cover
        """Re-implement to load derived-specific content from the archive.

        :meta public:
        """
        raise NotImplementedError

    @staticmethod
    def assert_equal(lhs: "BaseController", rhs: "BaseController") -> bool:
        """Re-implement for savable controllers to test for successful
        roundtrip"""
        raise NotImplementedError

    def save(self, *args, **kwargs) -> Path:
        """Easy access to global save function

        .. seealso:: `~amaze.simu.controllers.control.save`
        """
        from .control import save

        return save(self, *args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Easy access to global load function

        .. seealso:: `~amaze.simu.controllers.control.load`
        """
        from .control import load

        return load(*args, **kwargs)
