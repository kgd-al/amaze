""" Contains wrapper code to make AMaze work smoothly with stable baselines 3
"""
from typing import Type

from stable_baselines3 import SAC, A2C, DQN, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm as _BaseAlgorithm

from amaze.extensions.sb3.callbacks import TensorboardCallback
from amaze.extensions.sb3.controller import (wrapped_sb3_model as _wrap,
                                             load_sb3_controller)
from amaze.extensions.sb3.maze_env import make_vec_maze_env, env_method
from amaze.simu.controllers.base import BaseController as _BaseController
from amaze.simu.controllers.control import (CONTROLLERS as __BASE_CONTROLLERS)

print("[kgd-debug] >>> sb3 is being imported <<<")

__SB3_CONTROLLERS: dict[Type[_BaseAlgorithm], Type[_BaseController]] = {}
for c in [SAC, A2C, DQN, PPO, TD3]:
    c_type = _wrap(c)
    __BASE_CONTROLLERS[c_type.__repr__()] = c_type
    __SB3_CONTROLLERS[c] = c_type


def sb3_controller(model_type: Type[_BaseAlgorithm], *args, **kwargs):
    return __SB3_CONTROLLERS[model_type](*args, **kwargs)

