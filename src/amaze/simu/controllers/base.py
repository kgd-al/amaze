from abc import ABC, abstractmethod
from typing import Optional

from amaze.simu.pos import Vec
from amaze.simu.robot import State, InputType, OutputType


class BaseController(ABC):
    simple = True  # Whether this controller has a state (table, ANN, ...)

    def __init__(self, a_type: OutputType):
        self.action_type = a_type

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

    def to_json(self) -> dict:
        return dict()

    @classmethod
    def from_json(cls, dct: dict) -> 'BaseController':
        print(f'[kgd-debug] from json called on base controller, type={cls}')
        return cls()

    # noinspection PyMethodMayBeStatic
    def inputs_type(self) -> Optional[InputType]: return None
    # noinspection PyMethodMayBeStatic
    def outputs_type(self) -> Optional[OutputType]: return None
    # noinspection PyMethodMayBeStatic
    def vision(self) -> Optional[int]: return None
