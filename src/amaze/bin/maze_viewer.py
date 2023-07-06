#!/usr/bin/env python3

import argparse
import logging
from dataclasses import dataclass

from PyQt5.QtWidgets import QApplication

from amaze.simu.env.maze import Maze
from amaze.visu.maze import MazeWidget
from amaze.visu.viewer import MainWindow


@dataclass
class Options:
    view: bool = False
    autostart: bool = True

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            "Maze", "Initial settings for maze generation")
        Maze.BuildData.populate_argparser(group)

        parser.add_argument("--view", dest="view", action="store_true",
                            help="Show maze on screen")
        parser.add_argument("--no-autostart", dest="autostart",
                            action="store_false",
                            help="Whether to autostart the evaluation")


def main():
    args = Options()
    parser = argparse.ArgumentParser(description="2D Maze environment")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    app = QApplication([])
    logging.basicConfig(level=logging.DEBUG)

    if args.view:
        window = MainWindow(args)
        window.show()

        window.reset()

        if args.autostart:
            window.start()

        return app.exec()

    else:
        maze = Maze.generate(Maze.BuildData.from_argparse(args))
        maze_w = MazeWidget(maze)
        maze_w.draw_to(f"tmp/samples/{maze.seed}.png")


if __name__ == '__main__':
    main()
