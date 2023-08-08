from pathlib import Path
from typing import Optional, Dict, Type

import torch
from gymnasium import Space
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import SAC, A2C, DQN, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file

from amaze.simu.controllers.base import BaseController
from amaze.simu.pos import Vec
from amaze.simu.robot import State, InputType, OutputType
from amaze.sb3.utils import IOMapper

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


class SB3Controller(BaseController):
    simple = False

    def __init__(self, model: BaseAlgorithm):
        super().__init__()
        self.policy = model.policy

        self._i_type = _i_types_mapping[len(model.observation_space.shape)]
        self._o_type = _o_types_mapping[model.action_space.__class__]
        self._vision = None if self._i_type is InputType.DISCRETE \
            else model.observation_space.shape[0]

        self._mapper = IOMapper(observation_space=model.observation_space,
                                action_space=model.action_space)

        print(f"[kgd-debug] policy={self.policy.__class__.__name__}"
              f" {self._i_type=} {self._o_type=} {self._vision=}")

    def __call__(self, inputs: State) -> Vec:
        return self._mapper.map_action(
            self.policy.predict(
                self._mapper.map_observation(inputs),
                deterministic=True)[0])

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

    def save(self):
        pass

    def restore(self, state):
        pass

    def inputs_type(self): return self._i_type
    def outputs_type(self): return self._o_type
    def vision(self): return self._vision

    @staticmethod
    def load(path, model_class: Optional[type[BaseAlgorithm]] = None):
        print("Loading", path)

        if model_class is None:
            path = Path(path)
            model_class_path = path.with_suffix(".class")
            if model_class_path.exists():
                with open(model_class_path, 'r') as f:
                    model_class = _classes[f.read().replace('\n', '')]

            else:
                data, params, pytorch_variables = load_from_zip_file(path)
                if "model_class" in data:
                    model_class = data["model_class"]

        if model_class is None:
            raise ValueError("No model class provided and no metadata found")

        model = model_class.load(path)

        return SB3Controller(model)
