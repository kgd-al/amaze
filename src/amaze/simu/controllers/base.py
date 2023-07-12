from abc import ABC, abstractmethod
from typing import Tuple

from amaze.simu.pos import Vec, Pos


class BaseController(ABC):
    @abstractmethod
    def __call__(self, pos: Pos, inputs) -> Vec:
        """
        Generate next move

        :param pos: Current absolute position (should not be used)
        :param inputs: Current local information

        :return: (dx,dy) in [-1,1]x[-1,1], the requested movement
        """

    # noinspection PyMethodMayBeStatic
    def details(self) -> dict:
        return {}

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def restore(self, state):
        pass

    def to_json(self) -> dict:
        return dict()

    @classmethod
    def from_json(cls, dct: dict) -> 'BaseController':
        print(f'[kgd-debug] from json called on base controller, type={cls}')
        return cls()
