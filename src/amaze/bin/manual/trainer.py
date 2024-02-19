#!/usr/bin/env python3

import argparse
import functools
import logging
import math
import pprint
import shutil
import sys
import time
from dataclasses import dataclass, fields
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import humanize
import numpy as np
import pandas as pd

from amaze.simu.controllers.control import (controller_factory, Controllers,
                                            dump)
from amaze.simu.controllers.tabular import TabularController
from amaze.simu.env.maze import Maze
from amaze.simu.robot import Robot
from amaze.simu.simulation import Simulation
from amaze.utils.tee import Tee
from amaze.visu import resources
from amaze.visu.plotters.stats import plot_stats
from amaze.visu.plotters.tabular import plot_inputs_values
from amaze.visu.widgets.maze import MazeWidget

logger = logging.getLogger(__name__)


class TrainingType(str, Enum):
    SARSA = auto()
    Q_LEARNING = auto()


_controller_types = {
    TrainingType.SARSA: Controllers.TABULAR,
    TrainingType.Q_LEARNING: Controllers.TABULAR,
}


class OverwriteModes(str, Enum):
    ABORT = auto()
    IGNORE = auto()
    PURGE = auto()


@dataclass
class Options:
    id: Optional[str] = None
    base_folder: Path = Path("tmp")
    run_folder: Path = None  # automatically filled in
    overwrite: OverwriteModes = OverwriteModes.ABORT
    verbosity: int = 0
    quietness: int = 0

    training: TrainingType = None
    episodes = 10
    snapshots = 0
    mazes = []
    test_mazes = []

    seed: int = None
    alpha: float = .1
    gamma: float = .5

    epsilon: float = .3
    epsilon_decay: float = .001

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("--id", dest="id", metavar="ID",
                            help="Identifier for the run (and seed if integer"
                                 "and seed is not provided)")

        parser.add_argument("-f", "--folder",
                            dest="base_folder",
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

        parser.add_argument("--trainer", dest="training",
                            choices=[e.name for e in TrainingType],
                            type=str.upper, metavar="T",
                            required=True,
                            help="Which type of training will occur")

        parser.add_argument("--episodes", dest="episodes", metavar="E",
                            type=int, help="Number of training iterations")

        parser.add_argument("--snapshots", metavar='S', type=int,
                            help="Number of intermediate logging steps")

        group = parser.add_argument_group(
            "Reinforcement", "Settings for all reinforcement learning types")
        group.add_argument("--seed", dest="seed", help="Seed for RNG",
                           type=int, metavar="S")
        group.add_argument("--alpha", dest="alpha", help="Learning rate",
                           type=float, metavar="A")
        group.add_argument("--gamma", dest="gamma", help="Discount rate",
                           type=float, metavar="G")

        group = parser.add_argument_group(
            "Tabular", "Settings for tabular training (SARSA, Q-Learning)")
        group.add_argument("--epsilon", dest="epsilon", type=float,
                           help="Exploration probability", metavar="E")
        group.add_argument("--epsilon-decay", dest="epsilon", type=float,
                           help="Episodic reduction of epsilon", metavar="dE")

        group = parser.add_argument_group(
            "Maze", "Initial settings for maze generation")
        Maze.BuildData.populate_argparser(group)
        group.add_argument('--maze', metavar='M', dest='mazes',
                           action='append',
                           help="Full maze description (file or name)."
                                " Repeat to evaluate on multiple mazes"
                                " (sequentially). Arguments of the form"
                                " --maze-* will alter values for all mazes"
                                " (e.g. to set a common seed)")

        group.add_argument('--test-maze', metavar='M', dest='test_mazes',
                           action='append',
                           help="Full maze description. Works as 'maze' but"
                                " only for intermediate testing. The special"
                                " value 'train' adds all training mazes to"
                                " this list (which is also the default)."
                                " No overriding arguments are processed")
        parser.epilog += (
            f"\n"
            f"Sign arguments (cues and traps):\n"
            f"Signs are regular grayscale images."
            f" User-provided files are required to be squarish to ensure"
            f" correct aspect ratio, grayscale transformation will be applied"
            f" as needed. In addition the following library of built-in"
            f" resources is available: \n  {', '.join(resources.builtins())}")

        group = parser.add_argument_group(
            "Robot", "Robot settings")
        Robot.BuildData.populate_argparser(group)

    def normalize(self):

        self.verbosity -= self.quietness

        # Generate id if needed
        id_needed = (self.id is None)
        if id_needed:
            self.id = time.strftime('%Y%m%d%H%M%S')

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
            60, f"Using log level of"
                f" {logging.getLevelName(logger.root.level)}")
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
        # options.threads = max(1, min(options.threads,
        #                              len(os.sched_getaffinity(0))))
        # logger.info(f"Parallel: {options.threads}")

        self.training = TrainingType[self.training]

        if self.verbosity >= 0:
            raw_dict = {f.name: getattr(self, f.name) for f in fields(self)}
            logger.info(f"Post-processed command line arguments:"
                        f"\n{pprint.pformat(raw_dict)}")

        return log_level, tee


def bellman_train(simulation: Simulation, policy: TabularController, train):
    state = simulation.generate_inputs().copy()
    action = policy(state)

    while not simulation.done():
        reward = simulation.step(action)
        state_ = simulation.observations.copy()
        action_ = policy(state)
        train(state, action, reward, state_, action_)
        state, action = state_, action_

    return simulation.robot.reward


def greedy_eval(prefix, simulation, policy, mazes, args):
    p_state = policy.save()
    policy.epsilon = 0

    folder = args.run_folder
    rewards = []

    for i, maze in enumerate(mazes):
        prefix_ = f"{prefix}_{i}_"
        trajectory = []

        simulation.reset(maze=maze)
        action = policy(simulation.observations)
        while not simulation.done():
            pos = tuple(simulation.robot.pos)
            reward = simulation.step(action)
            trajectory.append((pos, action, reward))
            action = policy(simulation.observations)
        rewards.append(simulation.robot.reward)

        if args.verbosity >= 2:
            with open(folder.joinpath(prefix_ + "trajectory.dat"), 'w') as f:
                for (x, y), (i_, j_), r in trajectory:
                    f.write(f"{x} {y} {i_:+g} {j_:+g} {r:+.3g}\n")

        mw = MazeWidget()
        mw.set_simulation(simulation)
        mw.plot_trajectory(folder.joinpath(prefix_ + "trajectory.png"),
                           trajectory)

    plot_inputs_values(policy, str(folder.joinpath(f"{prefix}_iv.png")))

    policy.restore(p_state)
    if len(mazes) == 1:
        logger.info(f"Test eval, reward={rewards[0]}")
        return rewards[0], rewards[0], 0

    else:
        def pretty(lst): return "[" + ", ".join(f"{v:+.2g}" for v in lst) + "]"
        avg, std = np.average(rewards), np.std(rewards)
        logger.info(f"Test eval {prefix}, rewards={pretty(rewards)}"
                    f" (avg={avg:.2g}, std={std:.2g})")
        return rewards, avg, std


def process_mazes(args):
    if len(args.mazes) == 0:
        mazes = [Maze.generate(Maze.BuildData.from_argparse(args))]
    else:
        overrides = Maze.BuildData.from_argparse(args, set_defaults=False)
        mazes = [Maze.from_string(s, overrides) for s in args.mazes]

    maze_names = [m.to_string() for m in mazes]
    mazes_list = "".join(["\n> " + n for n in maze_names])
    assert len(set(maze_names)) == len(maze_names), \
        f"Duplicate mazes are not allowed:{mazes_list}"

    if len(args.test_makers) == 0:
        test_mazes = mazes
    else:
        test_mazes = [Maze.from_string(s) for s in args.train_mazes]

    maze_names = [m.to_string() for m in test_mazes]
    mazes_list = "".join(["\n> " + n for n in maze_names])
    assert len(set(maze_names)) == len(maze_names), \
        f"Duplicate mazes are not allowed:{mazes_list}"

    with open(args.run_folder.joinpath("mazes_train.dat"), 'w') as f:
        f.write(mazes_list.replace("> ", '') + "\n")
    logger.info(f"Training with maze(s):{mazes_list}")

    mazes_paths = []
    for i, maze in enumerate(mazes):
        path = args.run_folder.joinpath(f"train_{i}.maze")
        mazes_paths.append(path)
        maze.save(path)

    with open(args.run_folder.joinpath("mazes_test.dat"), 'w') as f:
        f.write(mazes_list.replace("> ", '') + "\n")
    logger.info(f"Testing with maze(s):{mazes_list}")

    for i, maze in enumerate(test_mazes):
        path: Path = args.run_folder.joinpath(f"test_{i}.maze")
        if mazes[i] == maze:
            if not path.exists():
                path.symlink_to(mazes_paths[i])
        else:
            maze.save(path)

    return mazes, test_mazes


def main():
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

    snapshot_steps = args.episodes / args.snapshots \
        if args.snapshots > 0 else 0

    train_mazes, test_mazes = process_mazes(args)

    robot = Robot.BuildData.from_argparse(args)

    # if True:
    #     # noinspection PyUnusedLocal
    #     app = QApplication([])

    if args.training in [TrainingType.SARSA, TrainingType.Q_LEARNING]:
        logger.info(f" Maze: {Maze.BuildData.from_argparse(args)}")
        logger.info(f"Robot: {Robot.BuildData.from_argparse(args)}")

        robot.control = _controller_types[args.training]
        robot.control_data = dict(
            actions=Simulation.discrete_actions(),
            epsilon=args.epsilon, seed=args.seed
        )
        policy: TabularController = controller_factory(
            robot.control, robot.control_data)
        train_function = {
            TrainingType.SARSA: policy.sarsa,
            TrainingType.Q_LEARNING: policy.q_learning
        }[args.training]
        train_function = functools.partial(
            train_function,
            alpha=args.alpha, gamma=args.gamma)
        simulation = Simulation(train_mazes[0], robot)

        stats = pd.DataFrame(columns=["R", "e"])
        test_stats = pd.DataFrame(
            columns=["Avg", "Std"] + [f"M{i}"
                                      for i in range(len(train_mazes))])

        s_name = None
        e_digits = math.ceil(math.log10(args.episodes))
        optimal = 0
        for e in range(args.episodes):
            m = train_mazes[e % len(train_mazes)]
            simulation.reset(m)
            reward = bellman_train(simulation, policy, train_function)

            if args.verbosity >= 1:
                logger.info(f"Episode {e:0{e_digits}d}, reward={reward},"
                            f" e={policy.epsilon}, maze={m.to_string()}")
            stats.loc[len(stats)] = [reward, policy.epsilon]

            if (e_ := policy.epsilon - args.epsilon_decay) > .01:
                policy.epsilon = e_

            if snapshot_steps > 0 and \
                    (e == 0 or (e % snapshot_steps) == (snapshot_steps - 1)):
                s_name = f"e{e:0{e_digits}}"
                rewards, avg_reward, std_rewards = \
                    greedy_eval(s_name, simulation, policy, test_mazes, args)
                dump(policy, args.run_folder.joinpath(s_name + ".ctrl"))
                test_stats.loc[e] = [avg_reward, std_rewards, *rewards]

                if math.isclose(avg_reward, Simulation.optimal_reward):
                    optimal += 1
                else:
                    optimal = 0

                if optimal >= 2:
                    logger.info("Early convergence. Breaking")
                    break

        # ======================================================================

        logger.debug("-- Aliasing final logs/images/... --")
        for i_file in sorted(args.run_folder.glob(f"{s_name}*")):
            o_file = i_file.with_stem(i_file.stem.replace(s_name, "final"))
            if not o_file.exists():
                o_file.symlink_to(i_file)
            logger.debug(f"{i_file} > {o_file}")

        # ======================================================================

        if args.verbosity >= 1:
            policy.pretty_print(show_updates=(args.verbosity >= 2))
        dump(policy, args.run_folder.joinpath("final.ctrl"))

        plot_stats(args.run_folder, stats, test_stats)

        # ======================================================================

    duration = humanize.precisedelta(
        timedelta(seconds=time.perf_counter() - start))
    logger.info(f"Completed training in {duration}")


if __name__ == '__main__':
    main()


"""
[X] Early stop on convergence?
Better epsilon decay
[?] Package requirements
[x] Log file
[ ] Individual plot line per maze?
"""
