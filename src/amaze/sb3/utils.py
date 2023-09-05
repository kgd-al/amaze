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

    QPA_PATH_NAME = "QT_QPA_PLATFORM_PLUGIN_PATH"
    QPA_PLATFORM_NAME = "QT_QPA_PLATFORM"

    def __init__(self, platform=True, path=True):
        self.qta_platform = platform
        self.qta_path = path

    @staticmethod
    def _save_and_replace(key, override):
        value = os.environ.get(key, None)
        os.environ[key] = override
        return value

    def __enter__(self):
        if self.qta_platform:
            self.qta_platform = self._save_and_replace(
                self.QPA_PLATFORM_NAME, "offscreen")
        if self.qta_path:
            self.qta_path = self._save_and_replace(
                self.QPA_PATH_NAME, QLibraryInfo.location(QLibraryInfo.PluginsPath))

    @staticmethod
    def _restore_or_clean(key, saved_value):
        if isinstance(saved_value, str):
            os.environ[key] = saved_value
        elif saved_value:
            os.environ.pop(key)

    def __exit__(self, *_):
        self._restore_or_clean(self.QPA_PLATFORM_NAME, self.qta_platform)
        self._restore_or_clean(self.QPA_PATH_NAME, self.qta_path)
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
