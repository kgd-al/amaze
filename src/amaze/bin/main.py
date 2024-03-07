#!/usr/bin/env python3
"""
Main executable for the AMaze library.

Provides and all-in-one entry point for simulation, evaluation and visualization
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Optional, Sequence

from amaze import application, load, Maze, Robot, Simulation, MazeWidget
from amaze.visu.viewer import MainWindow

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

    autostart: bool = True
    """ Whether to directly start moving the agent around """

    autoquit: bool = False
    """ Whether to close the viewer once the simulation is done """

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
        group = parser.add_argument_group(
            "Maze", "Initial settings for maze generation")
        Maze.BuildData.populate_argparser(group)

        group = parser.add_argument_group(
            "Robot", "Initial settings for robot configuration")
        Robot.BuildData.populate_argparser(group)

        parser.add_argument("--maze", dest="maze",
                            help="Use provided string-format maze")

        parser.add_argument("--auto-start", dest="autostart",
                            action="store_false",
                            help="Whether to autostart the evaluation")
        parser.add_argument("--no-auto-start", dest="autostart",
                            action="store_false",
                            help="see autostart")

        parser.add_argument("--auto-quit", dest="autoquit",
                            action="store_true",
                            help="Whether to quit after completing a maze")
        parser.add_argument("--no-auto-quit", dest="autoquit",
                            action="store_false",
                            help="see autoquit")

        parser.add_argument("--controller", dest="controller",
                            type=str, help="Load robot/controller from file"
                                           "or force use of a builtin"
                                           "(keyboard or random)")

        parser.add_argument("--robot-mode", dest="is_robot",
                            action="store_true",
                            help="See the maze as if you were a robot")

        parser.add_argument("--evaluate", dest="eval", type=Path,
                            help="Evaluate provided controller on provided"
                                 " maze and store results under the provided"
                                 " folder")

        parser.add_argument("--render", dest="render", type=Path,
                            nargs='?', const='maze.png',
                            help="Render maze to requested file")

        parser.add_argument("--movie", dest="movie", type=Path,
                            help="Render a video of the robot's strategy "
                                 "to requested file (gif)")

        parser.add_argument("--trajectory", dest="plot",
                            type=Path, nargs='?', const='trajectory.png',
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

        parser.add_argument("--extension", action='append',
                            dest='extensions',
                            help="Request extension specific controllers to be"
                                 "made available")


def __make_simulation(args, trajectory=False):
    if args.maze:
        maze_bd = Maze.BuildData.from_string(
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

    simulate = args.eval or (args.plot and not args.is_robot)
    window = not (args.render or simulate)
    if not window:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = application()

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
            if widget.render_to(args.render):
                logger.info(f"Saved {simulation.maze.to_string()}"
                            f" to {args.render}")

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
                    simulation=simulation, size=args.width, path=args.plot,
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
    main()
