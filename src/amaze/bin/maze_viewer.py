#!/usr/bin/env python3

import argparse
import logging
# TODO Careful
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QApplication

original_warn = warnings.warn
warnings.warn = lambda msg, category, *args, **kwargs: (
    original_warn(msg, category, *args, **kwargs)
    if category is not DeprecationWarning else None)


from amaze.simu.env.maze import Maze
from amaze.sb3.utils import CV2QTGuard
from amaze.visu.viewer import MainWindow


@dataclass
class Options:
    maze: Optional[str] = None
    controller: Optional[Path] = None
    autostart: bool = True
    render: Optional[Path] = None

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

        parser.add_argument("--controller", dest="controller", type=Path,
                            help="Load robot/controller from file")

        parser.add_argument("--render", dest="render", type=Path,
                            help="Render maze to requested file (and quit)")


def main():
    args = Options()
    parser = argparse.ArgumentParser(description="2D Maze environment")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    app = QApplication([])
    logging.basicConfig(level=logging.DEBUG)

    window = MainWindow(args)
    window.reset()

    if args.render:
        window.maze_w.update_config("robot", False)
        window.maze_w.update_config("dark", True)
        window.save_on_exit = False
        window.maze_w.draw_to(args.render)

    else:
        window.show()

        if args.autostart:
            window.start()

        return app.exec()


if __name__ == '__main__':
    with CV2QTGuard():
        main()
