#!/usr/bin/env python3

import argparse
import copy
import functools
import logging
import math
import pprint
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, fields, field
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union

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

logger = logging.getLogger("sb3-it-main")


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
    id: Optional[Union[str, int]] = None
    base_folder: Path = Path("tmp/sb3/")
    run_folder: Path = None  # automatically filled in
    overwrite: OverwriteModes = OverwriteModes.ABORT
    verbosity: int = 0
    quietness: int = 0

    trainer: str = None
    budget: int = 3_000_000
    trajectories: int = 10
    evals: int = 100

    all_permutations: bool = True
    mazes_file: Path = None

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

        parser.add_argument("--mazes", dest='mazes_file',
                            type=Path, help="Path for the set of mazes to use")

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
    mazes = [[], []]
    with open(args.mazes_file, 'r') as f:
        f.readline()  # Discard header
        while line := f.readline():
            s_id, m_str = line.replace('\n', '').split(' ')[:2]
            train = (s_id == '0')
            mazes[train].append(all_permutations([Maze.bd_from_string(m_str)]))
    pprint.pprint(mazes)
    train_mazes, test_mazes = mazes[True], mazes[False]

    assert len(train_mazes) == len(test_mazes)

    for name, maze_list in [("Train", train_mazes), ("Test", test_mazes)]:
        pretty_list = ""
        for i, m in enumerate(maze_list):
            pretty_list += f"== Stage {i} {'-' * 10}\n"
            pretty_list += ''.join(f"> {Maze.bd_to_string(bd)}\n" for bd in m)
        logger.info(f"{name}ing with maze(s):{pretty_list}")

    return train_mazes, test_mazes, len(train_mazes)


def agg(e, f_, f__): return f_(e.env_method(f__))


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    start = time.perf_counter()

    args = Options()
    parser = argparse.ArgumentParser(
        description="Main trainer for incremental mazes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n------------------\n"
               "Additional options\n"
    )
    Options.populate(parser)
    parser.parse_args(namespace=args)
    # noinspection PyUnusedLocal
    logging_level, tee = args.normalize()

    train_mazes_list, eval_mazes_list, stages = process_mazes(args)

    robot = Robot.BuildData.from_argparse(args)
    # robot = Robot.BuildData(inputs=InputType.DISCRETE,
    #                         outputs=OutputType.DISCRETE,
    #                         vision=36)
    logger.info(f"Using\n{pprint.pformat(robot)}")

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

    logger.info(f"Training with {trainer.__name__} supported by {policy}")
    run_folder = str(args.run_folder)

    stage_prefix = f"stage_{{:0{math.ceil(math.log10(len(train_mazes_list)))}}}"

    model = None
    remaining_budget = args.budget
    # time = 0

    logger.info(f"\n[{'=' * 35} Starting {'=' * 35}")

    for i, (train_mazes, eval_mazes) in \
            enumerate(zip(train_mazes_list, eval_mazes_list)):

        logging.root.handlers[0].formatter._fmt = (
            logging.root.handlers[0].formatter._fmt
            .replace('%(message)s', f"[Stage {i}] %(message)s"))

        logger.info(f"Remaining budget: {remaining_budget}")
        budget = round(remaining_budget / (stages - i))
        logger.info(f"Stage {i} budget: {budget} (base would be {args.budget / stages})")

        logger.info(f"Switching to stage {i} with environments:\n"
                    + ''.join(f"> {Maze.bd_to_string(bd)}\n" for bd in train_mazes)
                    + "===\n"
                    + ''.join(f"> {Maze.bd_to_string(bd)}\n" for bd in eval_mazes))

        stage_folder = str(args.run_folder.joinpath(f"stage{i}"))

        n_train_envs = len(train_mazes)
        n_eval_envs = len(eval_mazes)

        # Initialize a vectorized training environment with default parameters
        def env_fn(env_list: list, log_trajectory=False):
            env = MazeEnv(maze=env_list.pop(0), robot=robot,
                          log_trajectory=log_trajectory)
            check_env(env)
            env.reset(full_reset=True)
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

        # Periodically evaluate agent
        trajectories_freq = args.trajectories
        train_duration = agg(train_env, np.sum, "maximal_duration")
        if args.evals == -1:
            eval_freq = train_duration
            logger.info(f"Evaluating after every full training"
                        f" ({eval_freq} timesteps)")
        else:
            eval_freq = budget // args.evals
            if eval_freq < train_duration:
                eval_freq = train_duration
                logger.warning(f"Not enough budget for {args.evals} evaluations:\n"
                               f" train duration={train_duration}"
                               f" budget={budget}, evals={args.evals};"
                               f" {args.evals} > {budget / train_duration}\n"
                               f" Clamping to"
                               f" {int(budget // train_duration)}")
            else:
                logger.info(f"Evaluating every {eval_freq} call to env.step()"
                            f" (across {n_train_envs} evaluation environments)")
        optimal_reward = agg(eval_env, np.average, "optimal_reward")
        logger.info(f"Training will stop upon reaching average reward of"
                    f" {optimal_reward}")
        tb_callback = TensorboardCallback(
            log_trajectory_every=trajectories_freq,
            max_timestep=args.budget,
            prefix=stage_prefix.format(i), multi_env=True
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=stage_folder, log_path=stage_folder,
            eval_freq=max(1, eval_freq // n_eval_envs),
            n_eval_episodes=n_eval_envs,
            deterministic=True, render=False, verbose=args.verbosity,
            callback_after_eval=tb_callback,
            callback_on_new_best=StopTrainingOnRewardThreshold(
                reward_threshold=optimal_reward,
                verbose=1)
        )

        if model is None:
            model = trainer(policy, train_env,
                            seed=args.seed, learning_rate=1e-3)
        else:
            model.set_env(train_env)

        # Store the policy type for agnostic reload
        setattr(model, "model_class", model.__class__)
        with open(args.run_folder.joinpath("best_model.class"), 'wt') as f:
            f.write(model.__class__.__name__ + "\n")

        model.set_logger(configure(run_folder,
                                   ["stdout", "csv", "tensorboard"]))
        model.learn(budget, callback=eval_callback, progress_bar=True,
                    reset_num_timesteps=False)

        logger.info(f"[Stage {i}] Performing final logging step manually")
        tb_callback.log_step(True)

        remaining_budget = args.budget - model.num_timesteps

    msg = ""
    if model.num_timesteps < args.budget:
        msg += (f"Training converged in {model.num_timesteps}"
                f" / {args.budget} time steps)")
    else:
        msg += f"Training completed in {model.num_timesteps}"
    logger.info(f"{msg}.")

    # =========================================================================

    duration = humanize.precisedelta(
        timedelta(seconds=time.perf_counter() - start))
    logger.info(f"Completed training in {duration}")
    logger.info(f"Results are under {args.run_folder}")


if __name__ == "__main__":
    main()
