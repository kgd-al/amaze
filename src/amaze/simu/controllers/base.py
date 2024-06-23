from abc import ABC, abstractmethod
from typing import Optional, List
from zipfile import ZipFile

from amaze.simu.pos import Vec
from amaze.simu.types import InputType, OutputType, State


class BaseController(ABC):
    simple = True  # Whether this controller has no state (table, ANN, ...)

    infos: dict = {}
    """ Generic storage for additional information.

    For instance the class of mazes this agent should solve.
    """

    def __init__(self,
                 input_type: InputType, output_type: OutputType,
                 vision: Optional[int]):
        self._input_type = input_type
        self._output_type = output_type
        self._vision = vision

    @property
    def input_type(self) -> InputType: return self._input_type

    @property
    def output_type(self) -> OutputType: return self._output_type

    @property
    def vision(self) -> int: return self._vision

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

    def name(self): return self.__class__.__name__

    @abstractmethod
    def reset(self): pass

    @staticmethod
    @abstractmethod
    def inputs_types() -> List[InputType]:  # pragma no cover
        """ Specify what kind of inputs this controller can handle.

        Abstract method that should be implemented and documented.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def outputs_types() -> List[OutputType]:  # pragma no cover
        """ Specify what kind of outputs this controller can handle.

        Abstract method that should be implemented and documented.
        """
        raise NotImplementedError

    @abstractmethod
    def save_to_archive(self, archive: ZipFile, *args, **kwargs) \
            -> bool:  # pragma no cover
        """ Implement to save derived-specific content to the archive """
        raise NotImplementedError

    @classmethod
    def load_from_archive(cls, archive: ZipFile, *args, **kwargs) \
            -> 'BaseController':  # pragma no cover
        """ Implement to load derived-specific content from the archive """
        raise NotImplementedError
