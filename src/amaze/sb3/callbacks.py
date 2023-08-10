import logging
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import PIL.Image
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import Image, HParam, TensorBoardOutputFormat
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from amaze.sb3.graph import to_dot

logger = logging.getLogger(__name__)


class TensorboardCallback(BaseCallback):
    def __init__(self, log_trajectory_every: int = 0, verbose: int = 0,
                 max_timestep: int = 0):
        super().__init__(verbose=verbose)
        self.log_trajectory_every = log_trajectory_every
        fmt = "{:d}" if max_timestep == 0 \
            else "{:0" + str(math.ceil(math.log10(max_timestep-1))) + "d}"
        self.img_format = fmt

    def _on_training_start(self) -> None:
        policy = self.model.policy

        dd_rewards = defaultdict(list)
        for d in [e.__dict__ for e
                  in self.training_env.env_method('atomic_rewards')]:
            for key, value in d.items():
                dd_rewards[key].append(value)
        reward_strings = []
        for k, dl in dd_rewards.items():
            a, s = np.average(dl), np.std(dl)
            rs = f"{k[0]}={a:g}"
            if s != 0:
                rs += f"+/-{s:g}"
            reward_strings.append(rs)
        rewards_str = " ".join(reward_strings)

        io_types = set(
            i.name[0] + o.name[0] for i, o in
            self.training_env.env_method('io_types')
        )
        assert len(io_types) == 1, "Non-uniform I/O types"

        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "policy": policy.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "type": next(iter(io_types)),
            "rewards": rewards_str
        }

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

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here,
        # should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat))

        writer = self.tb_formatter.writer
        writer.add_text(
            "policy",
            str(policy).replace('\n', '<br/>')
            .replace(' ', '&nbsp;'))

        dummy_inputs = \
            policy.obs_to_tensor(policy.observation_space.sample())[0]
        writer.add_graph(policy, dummy_inputs, use_strict_trace=False)

        graph = to_dot(policy)
        graph.render(folder.joinpath("policy"), format='pdf', cleanup=True)
        graph.render(folder.joinpath("policy"), format='png', cleanup=True)
        # noinspection PyTypeChecker
        writer.add_image(
            "policy",
            np.asarray(PIL.Image.open(BytesIO(graph.pipe(format='jpg')))),
            dataformats="HWC", global_step=0)

    def _on_step(self) -> bool:
        self.log_step(False)
        return True

    def log_step(self, final: bool):
        logger.info(f"Logging tensorboard data at time"
                    f" {self.num_timesteps} ({final=})")

        assert isinstance(self.parent, EvalCallback)
        env = self.parent.eval_env

        for d in env.get_attr('last_infos'):
            for k, v in d.items():
                self.logger.record_mean("infos/" + k, v)

        print_trajectory = final or \
            (self.log_trajectory_every > 0
             and (self.n_calls % self.log_trajectory_every) == 0)
        if print_trajectory:
            t_str = "final" if final else\
                self.img_format.format(self.num_timesteps)
            images = env.env_method("plot_trajectory")

            big_image = tile_images(images)

            self.logger.record(f"infos/traj_{t_str}",
                               Image(big_image, "HWC"),
                               exclude=("stdout", "log", "json", "csv"))
            folder = Path(self.logger.dir).joinpath("trajectories")
            folder.mkdir(exist_ok=True)
            pil_img = PIL.Image.fromarray(big_image)
            pil_img.save(folder.joinpath(t_str + ".png"))

        self.logger.dump(self.model.num_timesteps)
