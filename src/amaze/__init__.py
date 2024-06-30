"""Root package for the AMaze benchmark generator
"""

from amaze.misc.resources import Sign
from amaze.misc.utils import qt_application
from .bin.main import main as amaze_main
from .simu import (
    Maze,
    Robot,
    Simulation,
    InputType,
    OutputType,
    StartLocation,
    load,
)
from .visu.widgets.maze import MazeWidget

__all__ = [
    "Maze",
    "Robot",
    "Simulation",
    "InputType",
    "OutputType",
    "StartLocation",
    "load",
    "MazeWidget",
    "qt_application",
    "Sign",
    "amaze_main",
]
