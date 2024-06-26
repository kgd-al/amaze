import math
from random import Random
from typing import List
from zipfile import ZipFile

from ..robot import Robot, InputType, OutputType
from amaze.simu.controllers.base import BaseController
from amaze.simu.types import Action


class RandomController(BaseController):
    _savable = False

    def __init__(self, robot_data: Robot.BuildData, seed=None):
        super().__init__(robot_data=robot_data)
        self.seed = seed
        self.rng = Random(seed)

    def __call__(self, _) -> Action:
        if self.output_type is OutputType.DISCRETE:
            return self.rng.choice(self.discrete_actions)
        else:
            return Action.from_polar(self.rng.random() * math.pi,
                                     self.rng.random())

    def reset(self):
        self.rng = Random(self.seed)

    def save_to_archive(self, archive: ZipFile, *args, **kwargs) \
            -> bool:  # pragma: no cover
        raise NotImplementedError

    def load_from_archive(self, archive: ZipFile, *args, **kwargs) \
            -> 'RandomController':  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def inputs_types() -> List[InputType]:
        return list(InputType)

    @staticmethod
    def outputs_types() -> List[OutputType]:
        return [OutputType.DISCRETE]
