from zipfile import ZipFile

from amaze.simu import InputType, OutputType
from amaze.simu.pos import Pos, Vec
from random import Random
from typing import Tuple, Optional, List

from amaze.simu.controllers.base import BaseController
from amaze.simu.maze import Maze


class CheaterController(BaseController):
    cheats = True

    def __init__(self, simulation, **kwargs):
        if simulation is None and not hasattr(self, "simulation"):
            raise ValueError("Cheater controller missing required 'simulation'"
                             " parameter")
        elif simulation is not None:
            self.simulation = simulation
        self.path = self.simulation.maze.solution[1:]
        self.current_cell = self.simulation.robot.pos.aligned()
        self.previous_cell = self.current_cell

    def __call__(self, _) -> Vec:
        self.current_cell = self.simulation.robot.pos
        if self.simulation.robot.data.outputs is OutputType.DISCRETE:
            return Vec(*self.path.pop(0)) - Vec(*self.current_cell.aligned())
        else:
            vec = (Vec(*self.path[0]) + Vec(.5, .5) - self.current_cell)
            vec /= vec.length()
            cell_index = self.current_cell.aligned()
            if cell_index != self.previous_cell:
                self.previous_cell = cell_index
                self.path.pop(0)
            return vec

    def reset(self):
        self.__init__(simulation=None)

    def save_to_archive(self, archive: ZipFile) -> bool:
        raise NotImplementedError

    def load_from_archive(self, archive: ZipFile) -> 'RandomController':
        raise NotImplementedError

    @staticmethod
    def inputs_types() -> List[InputType]:
        return list(InputType)

    @staticmethod
    def outputs_types() -> List[OutputType]:
        return list(OutputType)
