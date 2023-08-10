import os

import numpy as np
from PyQt5.QtCore import QLibraryInfo
from gymnasium import Space
from gymnasium.spaces import Discrete

from amaze.simu.pos import Vec
from amaze.simu.simulation import Simulation


class CV2QTGuard:
    """Acts as a guard allowing both PyQt5 and opencv-python to use the
     xcb.qpa plugin without confusion.

     Temporarily restores environmental variable "QT_QPA_PLATFORM_PLUGIN_PATH"
     to the value used by qt, taken from
     QLibraryInfo.location(QLibraryInfo.PluginsPath)
     """

    QPA_NAME = "QT_QPA_PLATFORM_PLUGIN_PATH"

    def __init__(self):
        self.qta_path = None

    def __enter__(self):
        if path := os.environ.get(self.QPA_NAME, None):
            self.qta_path = path
        os.environ[self.QPA_NAME] = \
            QLibraryInfo.location(QLibraryInfo.PluginsPath)

    def __exit__(self, *_):
        if self.qta_path:
            os.environ[self.QPA_NAME] = self.qta_path
        else:
            os.environ.pop(self.QPA_NAME)
        return False


class IOMapper:
    def __init__(self, observation_space: Space, action_space: Space):
        self.o_space = observation_space
        if len(self.o_space.shape) == 1:
            self.map_observation = lambda obs: obs
        else:
            self.map_observation = lambda obs: \
                (obs * 255) .astype(np.uint8).reshape(self.o_space.shape)

        self.a_space = action_space
        if isinstance(self.a_space, Discrete):
            self.action_mapping = Simulation.discrete_actions()
            self.map_action = lambda a: Vec(*self.action_mapping[a])
        else:
            self.map_action = lambda a: Vec(*a)
