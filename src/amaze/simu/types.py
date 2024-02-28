from enum import Enum

import numpy as np

from amaze.simu.pos import Vec


class InputType(Enum):
    """ Describes the type of input provided to the maze-navigating agent """

    DISCRETE = "DISCRETE"
    """ Input is made of 8 pre-processed floats"""

    CONTINUOUS = "CONTINUOUS"
    """ Input is a raw image representing the current cell, at a given
    resolution """


class OutputType(Enum):
    """ Describes how the agent is moving around in the maze"""

    DISCRETE = "DISCRETE"
    """ The agent moves from one cell to another """

    CONTINUOUS = "CONTINUOUS"
    """ The agent controls its acceleration """


Action = Vec
State = np.ndarray
