"""Root package for the AMaze benchmark generator
"""

from amaze.simu import Maze, Robot, Simulation, InputType, OutputType, load
from amaze.visu.widgets.maze import MazeWidget
from amaze.visu.widgets import application
from amaze.visu.resources import Sign
from amaze.bin.main import main as amaze_main
