from abc import ABC, abstractmethod
from typing import Optional
from zipfile import ZipFile

from amaze.simu.pos import Vec
from amaze.simu.types import InputType, OutputType, State


class BaseController(ABC):
    simple = True  # Whether this controller has no state (table, ANN, ...)

    infos: dict = {}
    """ Generic storage for additional information.
    
    For instance the class of mazes this agent should solve.
    """

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

    @abstractmethod
    def save(self):
        """Return a state from which self can be restored

         (i.e. to temporarily deactivate gamma exploration)"""
        pass

    @abstractmethod
    def restore(self, state):
        """Restore state from provided object"""
        pass

    @abstractmethod
    def save_to_archive(self, archive: ZipFile) -> bool:
        raise NotImplementedError

    @classmethod
    def load_from_archive(cls, archive: ZipFile) -> 'BaseController':
        raise NotImplementedError

    @staticmethod
    def inputs_type() -> InputType:
        raise NotImplementedError

    @staticmethod
    def outputs_type() -> OutputType:
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def vision(self) -> Optional[int]: return None
