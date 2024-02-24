from enum import Enum

import numpy as np

from amaze.simu.pos import Vec


class InputType(Enum):
    DISCRETE = "DISCRETE"
    CONTINUOUS = "CONTINUOUS"


class OutputType(Enum):
    DISCRETE = "DISCRETE"
    CONTINUOUS = "CONTINUOUS"


Action = Vec
State = np.ndarray
