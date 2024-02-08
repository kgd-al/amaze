#!/usr/bin/env python3

import argparse
import logging
import os
import pprint
# TODO Careful
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Sequence

from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from amaze.simu.controllers.control import load
from amaze.simu.robot import Robot
from amaze.simu.simulation import Simulation
from amaze.visu.widgets.maze import MazeWidget

original_warn = warnings.warn
warnings.warn = lambda msg, category, *args, **kwargs: (
    original_warn(msg, category, *args, **kwargs)
    if category is not DeprecationWarning else None)


from amaze.simu.env.maze import Maze
from amaze.sb3.utils import CV2QTGuard
from amaze.visu.viewer import MainWindow


logger = logging.getLogger(__name__)


@dataclass
class Options:
    maze: Optional[str] = None
    controller: Optional[Path] = None

    eval: Optional[Path] = None

    autostart: bool = True

    render: Optional[Path] = None
    plot: Optional[Path] = None
    width: int = 256
    cell_width: int = None

    dark: bool = False
    colorblind: bool = False

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            "Maze", "Initial settings for maze generation")
        Maze.BuildData.populate_argparser(group)

        parser.add_argument("--maze", dest="maze",
                            help="Use provided string-format maze")
        parser.add_argument("--no-autostart", dest="autostart",
                            action="store_false",
                            help="Whether to autostart the evaluation")

        parser.add_argument("--controller", dest="controller",
                            type=Path, help="Load robot/controller from file")

        parser.add_argument("--evaluate", dest="eval", type=Path,
                            help="Evaluate provided controller on provided"
                                 " maze and store results under the provided"
                                 " folder")

        parser.add_argument("--render", dest="render", type=Path,
                            nargs='?', const='maze.png',
                            help="Render maze to requested file")

        parser.add_argument("--trajectory", dest="plot", type=Path,
                            nargs='?', const='trajectory.png',
                            help="Plot trajectory of provided agent to"
                                 " provided path")

        parser.add_argument("--width", type=int,
                            help="Offscreen rendering target width")
        parser.add_argument("--cell-width", type=int,
                            help="Offscreen rendering target width of a"
                                 " single cell")

        parser.add_argument("--dark", action='store_true',
                            help="Dark background?"
                                 " (identical to agent's perceptions)")

        parser.add_argument("--colorblind", action='store_true',
                            help="Use colorblind-friendly colormaps")


def __make_simulation(args, trajectory=False):
    if args.maze:
        maze_bd = Maze.bd_from_string(
            args.maze, Maze.BuildData.from_argparse(args, set_defaults=False))
    else:
        maze_bd = Maze.BuildData.from_argparse(args, set_defaults=True)

    if args.cell_width is not None:
        args.width = args.cell_width * maze_bd.width

    return Simulation(
        Maze.generate(maze_bd),
        Robot.BuildData.from_argparse(args),
        save_trajectory=trajectory
    )


def main(sys_args: Optional[Sequence[str]] = None):
    args = Options()
    parser = argparse.ArgumentParser(description="2D Maze environment")
    Options.populate(parser)
    parser.parse_args(args=sys_args, namespace=args)

    if args.eval and not args.controller:
        print("Cannot evaluate without a controller")
        exit(1)

    if args.plot and not args.controller:
        print("Cannot plot trajectory without a controller")
        exit(1)

    if not args.controller:
        args.autostart = False

    if args.eval:
        if args.render and len(args.render.parts) == 1:
            args.render = args.eval.joinpath(args.render)
        if args.plot and len(args.plot.parts) == 1:
            args.plot = args.eval.joinpath(args.plot)
        args.eval.mkdir(parents=True, exist_ok=True)

    simulate = args.eval or args.plot
    window = not (args.render or simulate)
    if not window:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication([])
    logging.basicConfig(level=logging.DEBUG)

    if not window:
        simulation = __make_simulation(args)

        if args.render:
            widget = MazeWidget(simulation,
                                config=dict(
                                    robot=False,
                                    solution=True,
                                    dark=args.dark,
                                    colorblind=args.colorblind
                                ),
                                width=args.width)
            if widget.draw_to(args.render):
                logger.info(f"Saved {simulation.maze.to_string()} to {args.render}")

        if simulate:
            simulation.reset(save_trajectory=True)
            controller = load(args.controller)
            while not simulation.done():
                simulation.step(controller(simulation.observations))
            reward = simulation.robot.reward
            print(f"Cumulative reward: {reward} "
                  f"{simulation.infos()['pretty_reward']}")
            if args.plot:
                MazeWidget.plot_trajectory(
                    simulation=simulation,
                    size=args.width, trajectory=simulation.trajectory,
                    config=dict(
                        solution=True,
                        robot=False,
                        dark=True
                    ),
                    path=args.plot
                )
                print(f"Plotted {args.controller}"
                      f" in {simulation.maze.to_string()}"
                      f" to {args.plot}")

    else:
        window = MainWindow(args)
        window.reset()

        window.show()

        if args.autostart:
            window.start()

        return app.exec()

    return 0


if __name__ == '__main__':
    with CV2QTGuard(platform=False):
        main()
