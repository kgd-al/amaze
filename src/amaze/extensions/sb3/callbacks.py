""" Contains an out-of-the-box exemple of verbose callback relying on
Tensorboard.

Provided as-is without *any* guarantee of functionality or fitness for a
particular purpose
"""

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict

import PIL.Image
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Image, HParam, TensorBoardOutputFormat
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from amaze.extensions.sb3.maze_env import env_method, env_attr

logger = logging.getLogger(__name__)


def _recurse_avg_dict(dicts: List[Dict], root_key=""):
    avg_dict = defaultdict(list)
    for d in dicts:
        for k, v in _recurse_dict(d, root_key):
            avg_dict[k].append(v)
    return {
        k: np.average(v) for k, v in avg_dict.items()
    }


def _recurse_dict(dct, root_key):
    for k, v in dct.items():
        current_key = f"{root_key}/{k}" if root_key else k
        if isinstance(v, dict):
            for k_, v_ in _recurse_dict(v, current_key):
                yield k_, v_
        else:
            yield current_key, v


class TensorboardCallback(BaseCallback):
    def __init__(self, log_trajectory_every: int = 0, verbose: int = 0,
                 max_timestep: int = 0, prefix: str = "",
                 multi_env: bool = False):
        super().__init__(verbose=verbose)
        self.log_trajectory_every = log_trajectory_every
        fmt = "{:d}" if max_timestep == 0 \
            else "{:0" + str(math.ceil(math.log10(max_timestep-1))) + "d}"
        if prefix:
            if not prefix.endswith("_"):
                prefix += "_"
            fmt = prefix + fmt
        self.prefix = prefix
        self.img_format = fmt
        self.multi_env = multi_env

        self.last_stats: Optional = None

    @staticmethod
    def _rewards(env):
        dd_rewards = defaultdict(list)
        for d in [e.__dict__ for e in env_method(env, 'atomic_rewards')]:
            for key, value in d.items():
                dd_rewards[key].append(value)
        reward_strings = []
        for k, dl in dd_rewards.items():
            a, s = np.average(dl), np.std(dl)
            rs = f"{k[0]}={a:g}"
            if s != 0:
                rs += f"+/-{s:g}"
            reward_strings.append(rs)
        return " ".join(reward_strings)

    def _on_training_start(self) -> None:
        assert isinstance(self.parent, EvalCallback)

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here,
        # should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat))

        writer = self.tb_formatter.writer
        io_types = set(
            i.name[0] + o.name[0] for i, o in
            env_method(self.training_env, 'io_types')
        )
        assert len(io_types) == 1, "Non-uniform I/O types"

        if self.multi_env:
            writer.add_text(f"train/rewards",
                            self.prefix + ": " + self._rewards(self.training_env))
            writer.add_text(f"eval/rewards",
                            self.prefix + ":" + self._rewards(self.parent.eval_env))

        if self.num_timesteps > 0:
            return

        policy = self.model.policy
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "policy": policy.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "type": next(iter(io_types)),
        }
        if not self.multi_env:
            hparam_dict["rewards"] = self._rewards(self.training_env)

        metric_dict = {
            "infos/pretty_reward": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

        logger.info(f"Policy: {policy}")
        folder = Path(self.logger.dir)
        folder.mkdir(exist_ok=True)
        with open(folder.joinpath("policy.str"), 'w') as f:
            f.write(str(policy) + "\n")

        writer.add_text(
            "policy",
            str(policy).replace('\n', '<br/>')
            .replace(' ', '&nbsp;'))

        # dummy_inputs = \
        #     policy.obs_to_tensor(policy.observation_space.sample())[0]
        # writer.add_graph(policy, dummy_inputs, use_strict_trace=False)
        #
        # graph = to_dot(policy)
        # graph.render(folder.joinpath("policy"), format='pdf', cleanup=True)
        # graph.render(folder.joinpath("policy"), format='png', cleanup=True)
        # # noinspection PyTypeChecker
        # writer.add_image(
        #     "policy",
        #     np.asarray(PIL.Image.open(BytesIO(graph.pipe(format='jpg')))),
        #     dataformats="HWC", global_step=0)

    def _on_step(self) -> bool:
        self.log_step(False)
        return True

    def _print_trajectory(self, env, key, name):
        images = env_method(env, "plot_trajectory",
                            verbose=True, cb_side=0, square=True)

        big_image = tile_images(images)

        self.logger.record(f"infos/{key}_traj",
                           Image(big_image, "HWC"),
                           exclude=("stdout", "log", "json", "csv"))
        folder = Path(self.logger.dir).joinpath("trajectories")
        folder.mkdir(exist_ok=True)
        pil_img = PIL.Image.fromarray(big_image)
        pil_img.save(folder.joinpath(f"{key}_{name}.png"))

    def log_step(self, final: bool):
        # logger.info(f"[kgd-debug] Logging tensorboard data at time"
        #             f" {self.num_timesteps} {self.model.num_timesteps} ({final=})")

        assert isinstance(self.parent, EvalCallback)
        env = self.parent.eval_env

        eval_infos = env_attr(env, 'last_infos')
        for key, value in _recurse_avg_dict(eval_infos, "infos").items():
            self.logger.record_mean(key, value)

        print_trajectory = \
            (final or (self.log_trajectory_every > 0
                       and (self.n_calls % self.log_trajectory_every) == 0))

        if print_trajectory:
            t_str = f"final" if final else \
                self.img_format.format(self.num_timesteps)
            self._print_trajectory(env, "eval", t_str)

        self.logger.dump(self.model.num_timesteps)

        if final:
            train_env = self.training_env
            env_method(train_env, 'log_trajectory', True)

            logger.info(f"Final log step. Storing performance on training env")
            r = evaluate_policy(model=self.model, env=train_env)

            env_method(train_env, 'log_trajectory', False)

            t_str = f"final" if final else \
                self.img_format.format(self.num_timesteps)
            self._print_trajectory(train_env, "train", t_str)

            eval_infos = _recurse_avg_dict(eval_infos, "eval")
            train_infos = _recurse_avg_dict(env_attr(train_env, 'last_infos'),
                                            "train")

            self.last_stats = {
                "train/reward": np.average(r),
                "eval/reward": self.parent.best_mean_reward,
            }
            self.last_stats.update(eval_infos)
            self.last_stats.update(train_infos)
