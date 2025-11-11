from enum import Enum, Flag, auto

import numpy as np

from .pos import Vec


class InputType(Enum):
    """Describes the type of input provided to the maze-navigating agent"""

    DISCRETE = "DISCRETE"
    """ Input is made of 8 pre-processed floats:

        * First walls and previous direction in direct order
          (EAST, NORTH, WEST, SOUTH)
        * Then signs (same order). Only one value will be above 0

        See the readme for examples.
    """

    CONTINUOUS = "CONTINUOUS"
    """ Input is a raw image representing the current cell, at a given
        resolution. Iteration order is

        .. code-block::

            for y in range(v):
                for x in range(v):

        where v is the agent retina size. See the readme for examples.
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


class MazeClass(Enum):
    """The various high-level classes of mazes"""

    TRIVIAL = 0b000
    """A maze without intersections"""

    SIMPLE = 0b001
    """A maze with only clues"""

    LURES = 0b011
    """A maze with clues and lures (but no traps)"""

    TRAPS = 0b101
    """A maze with clues and traps (but no lures)"""

    COMPLEX = 0b111
    """A maze with clues, traps and lures"""

    INVALID = 0xFF
    """An invalid maze type (e.g. one with intersections and no cues)"""


class MazeMetrics(Flag):
    """The various metrics one can extract from a maze."""

    SURPRISINGNESS = auto()
    """ The entropy of the mazes states, i.e. how likely to see different cells """

    DECEPTIVENESS = auto()
    """ The entropy of the mazes similar states, i.e. how likely to see the same
     corridor/intersection with only the cue to differentiate them """

    INSEPARABILITY = auto()
    """ The difference between signs (not implemented)"""

    ALL = SURPRISINGNESS | DECEPTIVENESS | INSEPARABILITY
    """ Shorthand """


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
