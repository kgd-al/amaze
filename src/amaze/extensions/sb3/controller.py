import io
from pathlib import Path
from typing import Optional, Dict, Type, TypeVar, Generic, get_args, Union, Tuple
from zipfile import ZipFile

import numpy as np
import torch
from gymnasium import Space
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import SAC, A2C, DQN, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file

from amaze.simu.controllers.base import BaseController
from amaze.simu.controllers.control import save, CONTROLLERS, load
from amaze.simu.pos import Vec
from amaze.simu.types import InputType, OutputType, State
from amaze.extensions.sb3.utils import IOMapper

_classes = {
    c.__name__: c for c in [
        SAC, A2C, DQN, PPO, TD3
    ]
}
_i_types_mapping: Dict[int, InputType] = {
    1: InputType.DISCRETE,
    2: InputType.CONTINUOUS
}
_o_types_mapping: Dict[Type[Space], OutputType] = {
    Discrete: OutputType.DISCRETE,
    Box: OutputType.CONTINUOUS
}

#
# class SB3Controller(BaseController):
#     simple = False
#
#     def __init__(self):
#         model = getattr(self, "model")
#         # model = model_type(*args, **kwargs)
#         self.model = model
#         self.policy = model.policy
#
#         self._i_type = _i_types_mapping[len(model.observation_space.shape)]
#         self._o_type = _o_types_mapping[model.action_space.__class__]
#         self._vision = None if self._i_type is InputType.DISCRETE \
#             else model.observation_space.shape[0]
#
#         self._mapper = IOMapper(observation_space=model.observation_space,
#                                 action_space=model.action_space)
#
#         # print(f"[kgd-debug] policy={self.policy.__class__.__name__}"
#         #       f" {self._i_type=} {self._o_type=} {self._vision=}")
#
#     def __repr__(self) -> str:
#         return f"SB3Controller[{self.model.__class__.__name__}]"
#
#     def __call__(self, inputs: State) -> Vec:
#         return self._mapper.map_action(
#             self.policy.predict(
#                 self._mapper.map_observation(inputs),
#                 deterministic=True)[0])
#
#     def value(self, inputs: State) -> float:
#         if isinstance(self._mapper.a_space, Discrete):
#             actions = range(self._mapper.a_space.n)
#         else:
#             raise NotImplementedError
#         obs, _ = self.policy.obs_to_tensor(self._mapper.map_observation(inputs))
#         _, log_prob, _ = self.policy.evaluate_actions(
#             obs, torch.Tensor(actions))
#
#         return log_prob
#
#     def reset(self): pass
#
#     def inputs_types(self): return [self._i_type]
#     def outputs_types(self): return [self._o_type]
#     def vision(self): return self._vision
#
#     def save(self, path: str, *args, **kwargs) -> None:
#         print("[kgd-debug] SB3 wants to save")
#         save(self, path, {"algo": type(self.model).__name__},
#              *args, **kwargs)
#
#     def save_to_archive(self, archive: ZipFile, *args, **kwargs) -> bool:
#         print("[kgd-debug] saving in archive")
#         buffer = io.BytesIO()
#         self.model.save(buffer, *args, **kwargs)
#         archive.writestr("sb3.zip", buffer.getvalue())
#         return True
#
#     @staticmethod
#     def load(path, model_class: Optional[type[BaseAlgorithm]] = None):
#         if model_class is None:
#             path = Path(path)
#             model_class_path = path.with_suffix(".class")
#             if model_class_path.exists():
#                 with open(model_class_path, 'r') as f:
#                     model_class = _classes[f.read().replace('\n', '')]
#
#             else:
#                 data, params, pytorch_variables = load_from_zip_file(path)
#                 if "model_class" in data:
#                     model_class = data["model_class"]
#
#         if model_class is None:
#             raise ValueError("No model class provided and no metadata found")
#
#         model = model_class.load(path)
#
#         return SB3Controller(model)
#
#
# def wrap(model_type: type[BaseAlgorithm], *args, **kwargs):
#     sb3_attributes = model_type.__dict__
#     amaze_attributes = SB3Controller.__dict__
#     collisions = (set(sb3_attributes.keys())
#                   .intersection(set(amaze_attributes.keys())))
#     collisions = set(c for c in collisions if not c.startswith('__'))
#     collisions.remove("_abc_impl")
#     if len(collisions) > 0:
#         raise ValueError("Conflicting attributes between sb3 and amaze:\n"
#                          + "".join(f"> {f}\n" for f in collisions))
#     attributes = {}
#     attributes.update(sb3_attributes)
#     attributes.update(amaze_attributes)
#     wrapped_type = type(f"WrappedController[SB3.{model_type.__name__}]",
#                         (model_type, SB3Controller),
#                         attributes)
#
#     def _wrapper_init(obj, *_args, **_kwargs):
#         model_type.__init__(obj, *_args, **_kwargs)
#
#     wrapped_type.__init__ = _wrapper_init
#     wrapper = wrapped_type(*args, **kwargs)
#     return wrapper


def wrapped_sb3_model(model_type: Type[BaseAlgorithm]):
    class SB3Controller(model_type, BaseController):
        simple = False

        _model_type = model_type

        def __init__(self, *_args, **_kwargs):
            super().__init__(*_args, **_kwargs)

            # print(f"[kgd-debug] policy={self.policy.__class__.__name__}"
            #       f" {self._i_type=} {self._o_type=} {self._vision=}")

        def _setup_model(self) -> None:
            super()._setup_model()
            self._i_type = _i_types_mapping[len(self.observation_space.shape)]
            self._o_type = _o_types_mapping[self.action_space.__class__]
            self._vision = None if self._i_type is InputType.DISCRETE \
                else self.observation_space.shape[0]

            self._mapper = IOMapper(observation_space=self.observation_space,
                                    action_space=self.action_space)

        @classmethod
        def __repr__(cls) -> str:
            return f"SB3Controller[{cls._model_type.__name__}]"

        def __call__(self, inputs: State) -> Vec:
            return self._mapper.map_action(
                self.policy.predict(
                    self._mapper.map_observation(inputs),
                    deterministic=True)[0])

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            # print("predict", f"{deterministic=}")
            return super().predict(observation, state, episode_start,
                                   deterministic)

        def value(self, inputs: State) -> float:
            if isinstance(self._mapper.a_space, Discrete):
                actions = range(self._mapper.a_space.n)
            else:
                raise NotImplementedError
            obs, _ = self.policy.obs_to_tensor(self._mapper.map_observation(inputs))
            _, log_prob, _ = self.policy.evaluate_actions(
                obs, torch.Tensor(actions))

            return log_prob

        def reset(self):
            pass

        def inputs_types(self):
            return [self._i_type]

        def outputs_types(self):
            return [self._o_type]

        def vision(self):
            return self._vision

        def save(self, path: str, *_args, **_kwargs) -> None:
            save(self, path,
                 infos={"algo": self._model_type.__name__},
                 *_args, **_kwargs)

        def save_to_archive(self, archive: ZipFile, *_args, **_kwargs) -> bool:
            buffer = io.BytesIO()
            self._model_type.save(self, buffer, *_args, **_kwargs)
            archive.writestr("sb3.zip", buffer.getvalue())
            return True

        @classmethod
        def load_from_archive(cls, archive: ZipFile, *_args, **_kwargs):
            buffer = io.BytesIO(archive.read("sb3.zip"))
            loaded_model = cls._model_type.load(buffer, *_args, **_kwargs)
            model = cls(policy=loaded_model.policy,
                        env=loaded_model.env,
                        device=loaded_model.device,
                        _init_setup_model=False)
            model.__dict__.update(loaded_model.__dict__)
            model._setup_model()
            return model

    return SB3Controller


def load_sb3_controller(path: str | Path):
    return load(path)
