from enum import Enum

import numpy as np

from .pos import Vec


class InputType(Enum):
    """Describes the type of input provided to the maze-navigating agent"""

    DISCRETE = "DISCRETE"
    """ Input is made of 8 pre-processed floats:
        - First walls and previous direction in direct order 
        (EAST, NORTH, WEST, SOUTH)
        - Then signs (same order). Only one value will be above 0
        
        See the readme for examples    
    """

    CONTINUOUS = "CONTINUOUS"
    """ Input is a raw image representing the current cell, at a given
    resolution:
     
        - Iteration order is 
        ``` for y in range(v):
                for x in range(v):
        ```
        
        See the readme for examples
     """


class OutputType(Enum):
    """Describes how the agent is moving around in the maze"""

    DISCRETE = "DISCRETE"
    """ The agent moves from one cell to another """

    CONTINUOUS = "CONTINUOUS"
    """ The agent controls its acceleration """


Action = Vec
State = np.ndarray


class StartLocation(int, Enum):
    """Describes which of the maze's corner to use as the starting position"""

    SOUTH_WEST = 0
    SOUTH_EAST = 1
    NORTH_EAST = 2
    NORTH_WEST = 3

    def shorthand(self):
        return "".join(s[0] for s in self.name.split("_"))

    @classmethod
    def from_shorthand(cls, short):
        return {
            "SW": cls.SOUTH_WEST,
            "SE": cls.SOUTH_EAST,
            "NE": cls.NORTH_EAST,
            "NW": cls.NORTH_WEST,
        }[short]


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
