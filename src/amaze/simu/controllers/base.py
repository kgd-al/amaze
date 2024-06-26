from abc import ABC, abstractmethod
from typing import List
from zipfile import ZipFile

from ..pos import Vec
from ..types import InputType, OutputType, State, Action, classproperty
from ..robot import Robot


class BaseController(ABC):
    _simple = True
    _cheats = False
    _savable = True

    infos: dict = {}
    """ Generic storage for additional information.

    For instance the class of mazes this agent should solve.
    """

    _discrete_actions = [Action(1, 0),
                         Action(0, 1),
                         Action(-1, 0),
                         Action(0, -1)]

    def __init__(self, robot_data: Robot.BuildData):
        self._robot_data = robot_data

    @classproperty
    def is_simple(cls) -> bool:
        """ Whether this controller has no internal state (table, ANN, ...)
        or solely used for demonstration and testing"""
        return cls._simple

    @classproperty
    def cheats(cls) -> bool:
        """ Whether this controller uses simulation-level information.

        Only meant for test controller *not* for trained agents

        .. see:: `~amaze.simu.controllers.CheaterController`
        """
        return cls._cheats

    @classproperty
    def savable(cls) -> bool:
        """ Whether this controller contains a savable state.

        If True, implies that `save_to_archive` and `load_from_archive` are
        implemented
        """
        return cls._savable

    @property
    def input_type(self) -> InputType: return self._robot_data.inputs

    @property
    def output_type(self) -> OutputType: return self._robot_data.outputs

    @property
    def vision(self) -> int: return self._robot_data.vision

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

    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def reset(self): pass  # pragma no cover

    @staticmethod
    @abstractmethod
    def inputs_types() -> List[InputType]:  # pragma: no cover
        """ Specify what kind of inputs this controller can handle.

        Abstract method that should be implemented and documented.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def outputs_types() -> List[OutputType]:  # pragma: no cover
        """ Specify what kind of outputs this controller can handle.

        Abstract method that should be implemented and documented.
        """
        raise NotImplementedError

    @classmethod
    @property
    def discrete_actions(cls):
        """ Specifies the set of discrete actions that can be taken by an agent
        """
        return cls._discrete_actions

    @abstractmethod
    def save_to_archive(self, archive: ZipFile, *args, **kwargs) \
            -> bool:  # pragma: no cover
        """ Implement to save derived-specific content to the archive """
        raise NotImplementedError

    @classmethod
    def load_from_archive(cls, archive: ZipFile, *args, **kwargs) \
            -> 'BaseController':  # pragma: no cover
        """ Implement to load derived-specific content from the archive """
        raise NotImplementedError
