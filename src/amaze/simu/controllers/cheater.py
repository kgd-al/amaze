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
        self.timestep = self.simulation.timestep
        self.previous_timestep = self.timestep - 1
        self.action = None

    def __call__(self, _) -> Vec:
        self.timestep = self.simulation.timestep

        if self.action is not None and self.timestep == self.previous_timestep:
            return self.action

        self.current_cell = self.simulation.robot.pos
        if self.simulation.robot.data.outputs is OutputType.DISCRETE:
            self.action = (Vec(*self.path.pop(0))
                           - Vec(*self.current_cell.aligned()))
        else:
            self.action = (Vec(*self.path[0])
                           + Vec(.5, .5)
                           - self.current_cell)
            self.action /= self.action.length()
            cell_index = self.current_cell.aligned()
            if cell_index != self.previous_cell:
                self.previous_cell = cell_index
                self.path.pop(0)

        self.previous_timestep = self.timestep

        return self.action

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
