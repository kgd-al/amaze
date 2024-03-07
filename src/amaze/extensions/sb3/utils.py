""" Various utility functions """

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Discrete

from amaze.simu.pos import Vec
from amaze.simu.simulation import Simulation


class IOMapper:
    """ Transform AMaze's inputs/outputs types to SB3 objects """
    def __init__(self, observation_space: Space, action_space: Space):
        self.o_space = observation_space
        if len(self.o_space.shape) == 1:
            self.map_observation = lambda obs: obs
        else:
            self.map_observation = lambda obs: \
                (obs * 255).astype(np.uint8).reshape(self.o_space.shape)

        self.a_space = action_space
        if isinstance(self.a_space, Discrete):
            self.action_mapping = Simulation.discrete_actions()
            self.map_action = lambda a: Vec(*self.action_mapping[a])
        else:
            self.map_action = lambda a: Vec(*a)
