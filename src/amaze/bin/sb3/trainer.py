#!/usr/bin/env python3

import argparse
import copy
import functools
import logging
import pprint
import shutil
import sys
import time
from dataclasses import dataclass, fields, field
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import humanize
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, A2C, DQN, PPO, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from amaze.sb3.callbacks import TensorboardCallback
from amaze.sb3.maze_env import MazeEnv
from amaze.simu.env.maze import Maze, StartLocation
from amaze.simu.robot import Robot, InputType, OutputType
from amaze.utils.tee import Tee
from amaze.visu import resources

logger = logging.getLogger("sb3-main")


class OverwriteModes(str, Enum):
    ABORT = auto()
    IGNORE = auto()
    PURGE = auto()


TRAINERS = {
    (InputType.DISCRETE, OutputType.DISCRETE): [
        A2C, DQN, PPO,
    ],
    # (InputType.DISCRETE, OutputType.CONTINUOUS): [
    (InputType.CONTINUOUS, OutputType.DISCRETE): [
        A2C, DQN, PPO
    ],
    (InputType.CONTINUOUS, OutputType.CONTINUOUS): [
        A2C,      PPO, SAC, TD3
    ]
}


@dataclass
class Options:
    id: Optional[str | int] = None
    base_folder: Path = Path("tmp/sb3/")
    run_folder: Path = None  # automatically filled in
    overwrite: OverwriteModes = OverwriteModes.ABORT
    verbosity: int = 0
    quietness: int = 0

    trainer: str = None
    budget: int = 50_000
    trajectories: int = 10
    evals: int = 100

    train_mazes: list[str] = field(default_factory=list)
    eval_mazes: list[str] = field(default_factory=list)
    all_permutations: bool = False

    seed: int = None

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("--id", dest="id", metavar="ID",
                            help="Identifier for the run (and seed if integer"
                                 "and seed is not provided)")

        parser.add_argument("-f", "--folder", dest="base_folder",
                            metavar="F", type=Path,
                            help="Base folder under which to store data")

        parser.add_argument("--overwrite", dest="overwrite",
                            choices=[o.name for o in OverwriteModes],
                            type=str.upper, metavar="O", nargs='?',
                            help="Purge target folder before started")

        parser.add_argument("-v", "--verbose", dest="verbosity",
                            action='count', help="Increase verbosity level")

        parser.add_argument("-q", "--quiet", dest="quietness",
                            action='count', help="Decrease verbosity level")

        parser.add_argument("--trainer", dest="trainer",
                            choices=list(set(v.__name__
                                             for vl in TRAINERS.values()
                                             for v in vl)),
                            type=str.upper, metavar="T",
                            required=True,
                            help="What RL algorithm to use")

        parser.add_argument("--budget", dest="budget", metavar="B",
                            type=int, help="Total number of timesteps")

        parser.add_argument("--trajectories", metavar='N', type=int,
                            help="Number of trajectories to log")

        parser.add_argument("--evals", metavar='E', type=int,
                            help="Number of intermediate evaluations")

        group = parser.add_argument_group(
            "Reinforcement", "Settings for all reinforcement learning types")
        group.add_argument("--seed", dest="seed", help="Seed for RNG",
                           type=int, metavar="S")

        group = parser.add_argument_group(
            "Maze", "Initial settings for maze generation")
        Maze.BuildData.populate_argparser(group)
        group.add_argument('--maze', metavar='M', dest='train_mazes',
                           action='append',
                           help="Full maze description (file or name)."
                                " Repeat to evaluate on multiple mazes"
                                " (sequentially). Arguments of the form"
                                " --maze-* will alter values for all mazes"
                                " (e.g. to set a common seed)")

        group.add_argument('--eval-maze', metavar='M', dest='eval_mazes',
                           action='append',
                           help="Full maze description. Works as 'maze' but"
                                " only for intermediate testing. The special"
                                " value 'train' adds all training mazes to"
                                " this list (which is also the default)."
                                " No overriding arguments are processed")

        group.add_argument('--all-permutations', action='store_true',
                           dest='all_permutations',
                           help="Expand the list of mazes (training and"
                                " evaluation) to contain all starting points")

        parser.epilog += \
            f"\n" \
            f"Sign arguments (cues and traps):\n" \
            f"Signs are regular grayscale images." \
            f" User-provided files are required to be squarish to ensure" \
            f" correct aspect ratio, grayscale transformation will be applied" \
            f" as needed. In addition the following library of built-in" \
            f" resources is available: \n  {', '.join(resources.builtins())}"

        group = parser.add_argument_group(
            "Robot", "Robot settings")
        Robot.BuildData.populate_argparser(group)

    def normalize(self):
        self.verbosity -= self.quietness

        # Generate id if needed
        id_needed = (self.id is None)
        if id_needed:
            if self.seed is None:
                self.id = time.strftime('%m%d%H%M%S')
            else:
                self.id = self.seed

        # Define the run folder
        folder_name = self.id
        if not isinstance(folder_name, str):
            folder_name = f"run{self.id}"
        self.run_folder = self.base_folder.joinpath(folder_name).resolve()

        if self.run_folder.exists():
            if self.overwrite is None:
                self.overwrite = OverwriteModes.PURGE
            if not isinstance(self.overwrite, OverwriteModes):
                self.overwrite = OverwriteModes[self.overwrite]
            if self.overwrite is OverwriteModes.IGNORE:
                logger.info(f"Ignoring contents of {self.run_folder}")

            elif self.overwrite is OverwriteModes.PURGE:
                shutil.rmtree(self.run_folder, ignore_errors=True)
                logger.warning(f"Purging contents of {self.run_folder},"
                               f" as requested")

            else:
                raise OSError("Target directory exists. Aborting")
        self.run_folder.mkdir(parents=True, exist_ok=True)

        tee = Tee(self.run_folder.joinpath("log"))

        log_level = 20-self.verbosity*10
        logging.basicConfig(level=log_level, force=True,
                            stream=sys.stdout,
                            format="[%(asctime)s|%(levelname)s|%(module)s]"
                                   " %(message)s",
                            datefmt="%Y-%m-%d|%H:%M:%S")
        logging.addLevelName(60, "KGD")
        logging.addLevelName(logging.WARNING, "WARN")
        logger.log(
            60, f"Using log level of {logging.getLevelName(logger.root.level)}")
        for m in ['matplotlib']:
            logger_ = logging.getLogger(m)
            logger_.setLevel(logging.WARNING)
            logger.info(f"Muting {logger_}")
        logger.setLevel(log_level-10)

        if id_needed:
            logger.info(f"Generated run id: {self.id}")
        logger.info(f"Run folder: {self.run_folder}")

        if self.seed is None:
            try:
                self.seed = int(self.id)
            except ValueError:
                self.seed = round(1000 * time.time())
            logger.info(f"Deduced seed: {self.seed}")

        # # Check the thread parameter
        # options.threads = max(1, min(options.threads, len(os.sched_getaffinity(0))))
        # logger.info(f"Parallel: {options.threads}")

        if self.verbosity >= 0:
            raw_dict = {f.name: getattr(self, f.name) for f in fields(self)}
            logger.info(f"Post-processed command line arguments:"
                        f"\n{pprint.pformat(raw_dict)}")

        return log_level, tee


def all_permutations(mazes: list[Maze.BuildData]):
    def start_from(m: Maze.BuildData, s: StartLocation):
        m_ = copy.deepcopy(m)
        m_.start = s
        return m_
    sl = [StartLocation.NORTH_WEST, StartLocation.NORTH_EAST,
          StartLocation.SOUTH_WEST, StartLocation.SOUTH_EAST]
    return [start_from(m, s) for s in sl for m in mazes]


def process_mazes(args):
    if len(args.train_mazes) == 0:
        t_mazes = [Maze.BuildData.from_argparse(args)]
    else:
        overrides = Maze.BuildData.from_argparse(args, set_defaults=False)
        t_mazes = [Maze.bd_from_string(s, overrides) for s in args.train_mazes]

    if args.all_permutations:
        t_mazes = all_permutations(t_mazes)

    maze_names = [Maze.bd_to_string(m) for m in t_mazes]
    mazes_list = "".join(["\n> " + n for n in maze_names])
    assert len(set(maze_names)) == len(maze_names), \
        f"Duplicate mazes are not allowed:{mazes_list}"
    logger.info(f"Training with maze(s):{mazes_list}")

    if len(args.eval_mazes) == 0:
        e_mazes = t_mazes.copy()
    else:
        e_mazes = [Maze.bd_from_string(s) for s in args.eval_mazes]
        if args.all_permutations:
            e_mazes = all_permutations(e_mazes)

    maze_names = [Maze.bd_to_string(m) for m in e_mazes]
    mazes_list = "".join(["\n> " + n for n in maze_names])
    assert len(set(maze_names)) == len(maze_names), \
        f"Duplicate mazes are not allowed:{mazes_list}"
    logger.info(f"Testing with maze(s):{mazes_list}")

    with open(args.run_folder.joinpath("mazes.dat"), 'w') as f:
        f.write("Train Desc\n")
        for i, ml in enumerate([t_mazes, e_mazes]):
            for m in ml:
                f.write(f"{i} {m}\n")

    return t_mazes, e_mazes


def main():
    # check_maze_env()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    # test_pygame_env()

    start = time.perf_counter()

    args = Options()
    parser = argparse.ArgumentParser(
        description="Main trainer for maze tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n------------------\n"
               "Additional options\n"
    )
    Options.populate(parser)
    parser.parse_args(namespace=args)
    # noinspection PyUnusedLocal
    logging_level, tee = args.normalize()

    run_folder = str(args.run_folder)

    train_mazes, eval_mazes = process_mazes(args)

    robot = Robot.BuildData.from_argparse(args)
    # robot = Robot.BuildData(inputs=InputType.DISCRETE,
    #                         outputs=OutputType.DISCRETE,
    #                         vision=36)
    logger.info(f"Using\n{pprint.pformat(robot)}")

    n_train_envs = len(train_mazes)
    n_eval_envs = len(eval_mazes)

    # Initialize a vectorized training environment with default parameters
    def env_fn(env_list: list, log_trajectory=False):
        env = MazeEnv(maze=env_list.pop(0), robot=robot,
                      log_trajectory=log_trajectory)
        check_env(env)
        return env

    train_env = make_vec_env(
        functools.partial(env_fn, env_list=train_mazes),
        n_envs=n_train_envs, seed=args.seed)
    logger.info(f"{train_env=}")

    # Separate evaluation env, with different parameters passed via env_kwargs
    # Eval environments can be vectorized to speed up evaluation.
    eval_env = make_vec_env(
        functools.partial(env_fn, env_list=eval_mazes, log_trajectory=True),
        n_envs=n_eval_envs, seed=args.seed)
    logger.info(f"{eval_env=}")

    def agg(f_, f__): return f_(eval_env.env_method(f__))
    def avg(f_): return agg(np.average, f_)

    # Periodically evaluate agent
    trajectories_freq = args.trajectories
    eval_freq = args.budget // (args.evals * n_eval_envs)
    if eval_freq < (avg_duration := agg(np.sum, "duration")):
        eval_freq = avg_duration
        logger.warning(f"Not enough budget for {args.evals} evaluations,"
                       f" clamping to"
                       f" {args.budget // (avg_duration * n_eval_envs)}")
    else:
        logger.info(f"Evaluating every {eval_freq} call to env.step()")
    tb_callback = TensorboardCallback(
        log_trajectory_every=trajectories_freq,
        max_timestep=args.budget
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_folder, log_path=run_folder,
        eval_freq=eval_freq, n_eval_episodes=n_eval_envs,
        deterministic=True, render=False, verbose=args.verbosity,
        callback_after_eval=tb_callback,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=avg("optimal_reward"),
            verbose=1)
    )

    trainer = next((t for t in TRAINERS[(robot.inputs, robot.outputs)]
                    if t.__name__.casefold() == args.trainer.casefold()),
                   None)
    if not trainer:
        raise ValueError(f"Requested trainer {args.trainer} is not valid"
                         f" for given {robot.inputs}->{robot.outputs}"
                         f" combination")
    policy = {
        InputType.DISCRETE: "MlpPolicy",
        InputType.CONTINUOUS: "CnnPolicy"
    }[robot.inputs]

    logger.info(f"\n{'=' * 35} Starting {'=' * 35}")
    logger.info(f"Training with {trainer.__name__} supported by {policy}")

    policy_kwargs = {}
    # policy_kwargs = dict(
    #     activation_fn=torch.nn.ReLU,
    #     net_arch=dict(pi=[32, 32], vf=[32, 32]))
    # policy_kwargs.update(dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=64),
    #     net_arch=[]
    # ))

    model = trainer(policy, train_env, seed=args.seed,
                    policy_kwargs=policy_kwargs, learning_rate=1e-3)

    # Store the policy type for agnostic reload
    setattr(model, "model_class", model.__class__)
    with open(args.run_folder.joinpath("best_model.class"), 'wt') as f:
        f.write(model.__class__.__name__ + "\n")

    model.set_logger(configure(run_folder,
                               ["stdout", "csv", "tensorboard"]))
    model.learn(args.budget, callback=eval_callback, progress_bar=True)

    msg = ""
    if model.num_timesteps < args.budget:
        msg += (f"Training converged in {model.num_timesteps}"
                f" / {args.budget} time steps)")
    else:
        msg += f"Training completed in {model.num_timesteps}"
    logger.info(f"{msg}. Performing final logging step manually")
    tb_callback.log_step(True)

    # =========================================================================

    resets = pd.DataFrame(
        columns=["Resets", "Length"],
        data=[(r, l)
              for env in [train_env, eval_env]
              for r, l in zip(*[env.get_attr(a) for a in ["resets", "length"]])],
        index=pd.MultiIndex.from_tuples([
            (name, env_name)
            for name, env in [("train", train_env), ("eval", eval_env)]
            for env_name in env.get_attr("name")
        ])
    )

    # Test envs are reset at the start AND end
    resets.loc["eval", "Resets"] = (resets.loc["eval", "Resets"] / 2).values

    logger.info(f"== Environments summary ==\n{resets}")
    #
    # plogger = logging.getLogger(logger.name + "-re-eval")
    #
    # obs = eval_env.reset()
    # step = 0
    # dones = np.full(n_eval_envs, False)
    # while True:
    #     action, _ = model.predict(obs, deterministic=True)
    #     # plogger.info(f"Step {step + 1}")
    #     # plogger.info(f"Action: {action}")
    #     obs, rewards, done, info = eval_env.step(action)
    #     # plogger.info(f"{obs=}\n{reward=}\n{done=}")
    #     dones |= done
    #     # plogger.info(f"{dones=}")
    #     eval_env.render("human")
    #     if all(dones):
    #         # Note that the VecEnv resets automatically
    #         # when a done signal is encountered
    #         plogger.info(f"Goal reached! {rewards=}")
    #         break
    #     step += 1

    duration = humanize.precisedelta(
        timedelta(seconds=time.perf_counter() - start))
    logger.info(f"Completed training in {duration}")


if __name__ == "__main__":
    main()


# TODOLIST
# TODO Print policy (need to understand the underlying ANN)
# TODO Reset stats print twice the usage for evals (because of?)
# ---- Log trajectories every n evals
# ---- Log pretty reward every eval
