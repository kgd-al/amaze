""" Contains wrapper code to make AMaze work smoothly with stable baselines 3
"""
from pathlib import Path
from typing import Type

from amaze.extensions.sb3.guard import CV2QTGuard

print("[kgd-debug] >>> sb3 is being imported <<<")

with CV2QTGuard():
    print("[kgd-debug] >>> guarding against cv2 <<<")

    from stable_baselines3 import SAC, A2C, DQN, PPO, TD3
    from stable_baselines3.common.base_class import \
        BaseAlgorithm as _BaseAlgorithm

    from amaze.extensions.sb3.callbacks import TensorboardCallback
    from amaze.extensions.sb3.controller import wrapped_sb3_model as _wrap
    from amaze.extensions.sb3.maze_env import make_vec_maze_env, env_method
    from amaze.simu.controllers.base import BaseController as _BaseController
    from amaze.simu.controllers.control import \
        CONTROLLERS as __BASE_CONTROLLERS, load


def compatible_models():
    """ Returns the list of SB3 models that can be used with this extension """
    return [SAC, A2C, DQN, PPO, TD3]


__SB3_CONTROLLERS: dict[Type[_BaseAlgorithm], Type[_BaseController]] = {}
for c in compatible_models():
    c_type = _wrap(c)
    __BASE_CONTROLLERS[c_type.__repr__()] = c_type
    __SB3_CONTROLLERS[c] = c_type


def sb3_controller(model_type: Type[_BaseAlgorithm], *args, **kwargs):
    """ Creates a wrapper for a stable baselines 3 model.

     :param model_type: the type of SB3 model to wrap (PPO, A2C, ...)
     :param args: positional arguments to pass to the model
     :param kwargs: keyword arguments to pass to the model
     """
    return __SB3_CONTROLLERS[model_type](*args, **kwargs)


def load_sb3_controller(path: str | Path):
    """ Loads a wrapper stable baselines 3 model from an archive.

    :warning: do not forget to specify use of this extension (sb3) when
      loading from an executable from the core library (e.g.
      :mod:`~amaze.bin.main`)
    """
    return load(path)


print("[kgd-debug] >>> sb3 is imported <<<")
