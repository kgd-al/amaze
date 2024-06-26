"""Root package for the AMaze benchmark generator
"""

from amaze.simu import (Maze, Robot, Simulation, InputType, OutputType,
                        StartLocation, load)
from amaze.visu.widgets.maze import MazeWidget
from amaze.visu.widgets import qt_application
from amaze.visu.resources import Sign
from amaze.bin.main import main as amaze_main

__all__ = [
    "Maze", "Robot", "Simulation", "InputType", "OutputType",
    "StartLocation", "load",
    "MazeWidget", "qt_application",
    "Sign",
    "amaze_main"
]
