"""Root package for the AMaze benchmark generator
"""

from .simu import (Maze, Robot, Simulation, InputType, OutputType,
                   StartLocation, load)
from amaze.misc.utils import qt_application
from amaze.misc.resources import Sign
from .visu.widgets.maze import MazeWidget
from .bin.main import main as amaze_main


__all__ = [
    "Maze", "Robot", "Simulation", "InputType", "OutputType",
    "StartLocation", "load",
    "MazeWidget", "qt_application",
    "Sign",
    "amaze_main"
]
