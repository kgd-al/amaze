#!/usr/bin/env python3
"""
Main executable for the AMaze library.

Provides and all-in-one entry point for simulation, evaluation and
 visualization
"""

import argparse
import logging
import pprint
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Optional, Sequence

from ..misc.utils import qt_application, qt_offscreen
from ..simu.controllers.control import load
from ..simu.maze import Maze
from ..simu.robot import Robot
from ..simu.simulation import Simulation
from ..visu.viewer import MainWindow
from ..visu.widgets.maze import MazeWidget

logger = logging.getLogger(__name__)


@dataclass
class Options:
    """
    Namespace containing all options for the AMaze main executable
    """

    maze: Optional[str] = None
    """ String description of the maze

    :see:
        :meth:`~amaze.simu.maze.Maze.to_string`
        :meth:`~amaze.simu.maze.Maze.from_string`
    """

    robot: Optional[str] = None
    """ String description of the robot's attributes

    :see:
        :meth:`~amaze.simu.robot.Robot.BuildData.to_string`
        :meth:`~amaze.simu.robot.Robot.BuildData.from_string`
    """

    controller: Optional[str] = None
    """ Path to a pre-trained controller or name of built-in.

     :see: :meth:`~amaze.simu.controllers.control.builtin_controllers`
     :note: if a pre-trained controller is provided, the extension should be
        .zip
     """

    # =====================

    is_robot: bool = False
    """ Activate "robot-mode" interface.

     No global view of the maze, only local perceptions and the instantaneous
     and cumulative rewards
     """

    eval: Optional[Path] = None
    """ Path under which to store evaluation results """

    eval_inputs: Optional[Path] = None
    """ Path under which to store input evaluation results """

    autostart: bool = True
    """ Whether to directly start moving the agent around """

    autoquit: bool = False
    """ Whether to close the viewer once the simulation is done """

    restore_config: bool = True
    """ Whether to load configuration (window position, maze, robot, ...)
     from the persistent cache """

    dt: Optional[float] = None
    """ Specifies the duration of a timestep when using the viewer"""

    # =====================

    movie: Optional[Path] = None
    """ Where to store the movie of the agent's trajectory """

    render: Optional[Path] = None
    """ Where to render the maze """

    plot: Optional[Path] = None
    """ Where to render the agent's trajectory """

    width: int = 256
    """ Width of images / videos """

    cell_width: int = None
    """ Width of a maze cell, to ensure readable graphics """

    # =====================

    dark: bool = False
    """ Whether to render a black on white maze (like the robots do) """

    colorblind: bool = False
    """ Whether to use a colorblind-friendly palette """

    # =====================

    extensions: list[str] = field(default_factory=list)

    # =====================

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Maze", "Initial settings for maze generation")
        Maze.BuildData.populate_argparser(group)

        group = parser.add_argument_group("Robot", "Initial settings for robot configuration")
        Robot.BuildData.populate_argparser(group)

        parser.add_argument("--maze", dest="maze", help="Use provided string-format maze")

        parser.add_argument(
            "--robot",
            dest="robot",
            help="Use provided string-format robot details",
        )

        parser.add_argument(
            "--auto-start",
            dest="autostart",
            action="store_false",
            help="Whether to autostart the evaluation",
        )
        parser.add_argument(
            "--no-auto-start",
            dest="autostart",
            action="store_false",
            help="see autostart",
        )

        parser.add_argument(
            "--auto-quit",
            dest="autoquit",
            action="store_true",
            help="Whether to quit after completing a maze",
        )
        parser.add_argument(
            "--no-auto-quit",
            dest="autoquit",
            action="store_false",
            help="see autoquit",
        )

        parser.add_argument(
            "--dt",
            dest="dt",
            type=float,
            help="Specifies the duration of one timestep when" " using the viewer",
        )

        parser.add_argument(
            "--no-restore-config",
            dest="restore_config",
            action="store_false",
            help="Prevents configuration load from persistent" " cache",
        )

        parser.add_argument(
            "--controller",
            dest="controller",
            type=str,
            help="Load robot/controller from file"
            "or force use of a builtin"
            "(keyboard or random)",
        )

        parser.add_argument(
            "--robot-mode",
            dest="is_robot",
            action="store_true",
            help="See the maze as if you were a robot",
        )

        parser.add_argument(
            "--evaluate",
            dest="eval",
            type=Path,
            help="Evaluate provided controller on provided"
            " maze and store results under the provided"
            " folder",
        )

        parser.add_argument(
            "--evaluate-inputs",
            dest="eval_inputs",
            type=Path,
            help="Evaluate provided controller on all possible"
            " inputs according to the provided maze's"
            " signs and store the results under the"
            " provided folder",
        )

        parser.add_argument(
            "--render",
            dest="render",
            type=Path,
            nargs="?",
            const="maze.png",
            help="Render maze to requested file",
        )

        parser.add_argument(
            "--movie",
            dest="movie",
            type=Path,
            help="Render a video of the robot's strategy " "to requested file (gif)",
        )

        parser.add_argument(
            "--trajectory",
            dest="plot",
            type=Path,
            nargs="?",
            const="trajectory.png",
            help="Plot trajectory of provided agent to" " provided path",
        )

        parser.add_argument(
            "--width",
            type=int,
            default=None,
            help="Window or offscreen rendering target width",
        )
        parser.add_argument(
            "--cell-width",
            type=int,
            help="Offscreen rendering target width of a" " single cell",
        )

        parser.add_argument(
            "--dark",
            action="store_true",
            help="Dark background?" " (identical to agent's perceptions)",
        )
        parser.add_argument(
            "--colorblind",
            action="store_true",
            help="Use colorblind-friendly colormaps",
        )

        parser.add_argument(
            "--extension",
            action="append",
            dest="extensions",
            help="Request extension specific controllers to be" "made available",
        )


def __make_simulation(args, trajectory=False):
    maze = __make_maze(args)

    if args.cell_width is not None:
        args.width = args.cell_width * maze.width

    if args.robot:
        robot = Robot.BuildData.from_string(args.robot).override_with(
            Robot.BuildData.from_argparse(args, set_defaults=False)
        )
    else:
        robot = Robot.BuildData.from_argparse(args, set_defaults=True)
    controller = __make_controller(args, robot)

    return Simulation(maze, robot, save_trajectory=trajectory), controller


def __make_maze(args):
    if args.maze:
        maze_bd = Maze.BuildData.from_string(
            args.maze, Maze.BuildData.from_argparse(args, set_defaults=False)
        )
    else:
        maze_bd = Maze.BuildData.from_argparse(args, set_defaults=True)

    return Maze.generate(maze_bd)


def __make_controller(args, robot: Optional[Robot.BuildData] = None):
    controller = None
    if args.controller:
        controller = load(args.controller)
        if robot:
            robot.override_with(Robot.BuildData.from_controller(controller))

    return controller


def main(sys_args: Optional[Sequence[str] | str] = None):
    """
    Main function for the AMaze executable. Allows delegate call.
    """

    if isinstance(sys_args, str):
        sys_args = sys_args.split()

    args = Options()
    parser = argparse.ArgumentParser(description="2D Maze environment")
    Options.populate(parser)
    parser.parse_args(args=sys_args, namespace=args)

    if args.eval and not args.controller:
        print("Cannot evaluate without a controller")
        exit(1)

    if args.plot and not args.controller and not args.is_robot:
        print("Cannot plot trajectory without a controller")
        exit(1)

    if args.extensions:
        for m in args.extensions:
            import_module("amaze.extensions." + m)

    if not args.controller:
        args.autostart = False
    else:
        if args.controller.endswith(".zip"):
            p = Path(args.controller)
            if not p.exists():
                raise ValueError(f"No controller archive found for '{p}'")
            args.controller = p

    if args.movie:
        args.autostart = True

    if args.eval:
        if args.render and len(args.render.parts) == 1:
            args.render = args.eval.joinpath(args.render)
        if args.plot and len(args.plot.parts) == 1:
            args.plot = args.eval.joinpath(args.plot)
        args.eval.mkdir(parents=True, exist_ok=True)

    if args.eval_inputs:
        args.eval_inputs.mkdir(parents=True, exist_ok=True)

    simulate = args.eval or (args.plot and not args.is_robot)
    window = not (args.render or args.eval_inputs or simulate)
    if not window:
        qt_offscreen()

    app = qt_application()

    logging.basicConfig(level=logging.DEBUG)

    if args.eval_inputs and (controller := __make_controller(args)):
        res = Simulation.inputs_evaluation(args.eval_inputs, controller, __make_maze(args).signs)

        pprint.pprint(res)

    if not window:
        simulation, controller = __make_simulation(args)

        if args.render:
            widget = MazeWidget.from_simulation(
                simulation,
                config=dict(
                    robot=False,
                    solution=True,
                    dark=args.dark,
                    colorblind=args.colorblind,
                ),
            )
            if widget.render_to_file(args.render, width=args.width):
                logger.info(f"Saved {simulation.maze.to_string()}" f" to {args.render}")

        if simulate:
            simulation.reset(save_trajectory=True)
            while not simulation.done():
                simulation.step(controller(simulation.observations))
            reward = simulation.robot.reward
            print(f"Cumulative reward: {reward} " f"{simulation.infos()['pretty_reward']}")
            if args.plot:
                MazeWidget.plot_trajectory(
                    simulation=simulation,
                    size=args.width,
                    path=args.plot,
                )
                print(
                    f"Plotted {args.controller}"
                    f" in {simulation.maze.to_string()}"
                    f" to {args.plot}"
                )

    else:
        window = MainWindow(args)
        window.reset()

        window.show()

        if args.autostart:
            window.start()

        return app.exec()

    return 0


if __name__ == "__main__":
    main()
