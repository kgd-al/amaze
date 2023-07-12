import logging

import gymnasium as gym
from gymnasium import spaces

from amaze.simu.simulation import Simulation


logger = logging.getLogger(__name__)

class MazeEnv(gym.Env):
    metadata = dict(
        render_modes=["human"],
        render_fps=30
    )

    def __init__(self, *args, **kwargs):
        logger.debug("Creating MazeEnv")
        super().__init__()
        self._simulation = Simulation()

        # self.action_space = spaces.
