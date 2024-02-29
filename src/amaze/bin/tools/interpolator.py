#!/usr/bin/env python3
import argparse
import json
import math
import pprint
from dataclasses import dataclass
from pathlib import Path

from amaze.simu.maze import Maze
from amaze.simu.types import InputType
from amaze.simu.simulation import Simulation
from amaze.visu.resources import Sign
from amaze.visu.widgets.maze import MazeWidget


@dataclass
class Options:
    maze_0: str = ""
    maze_1: str = ""
    mazes: int = 0
    rules: Path = Path()

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("-0", dest="maze_0", required=True,
                            help="String-format maze for u=0")
        parser.add_argument("-1", dest="maze_1", required=True,
                            help="String-format maze for u=1")
        parser.add_argument("-n", dest="mazes", required=True,
                            type=int, help="Number of mazes to interpolate")
        parser.add_argument("--rules", dest="rules", required=True,
                            type=Path,
                            help="Path to the elementary interpolation rules")


def main():
    """ Interpolate between two mazes to generate a number of intermediates
    according to some rules"""

    def value(_v): return Sign(value=_v)

    def linear(_u, _v0, _v1): return _u * (_v1 - _v0) + _v0

    args = Options()
    parser = argparse.ArgumentParser(description="2D Maze environment")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    bd_0 = Maze.bd_from_string(args.maze_0)
    bd_1 = Maze.bd_from_string(args.maze_1)

    rules = {}
    with open(args.rules, 'r') as f:
        for k, v in json.load(f).items():
            for k_ in k.lower().split(","):
                rules[k_] = v
    pprint.pprint(rules)

    mazes = []
    m_id_digits = math.ceil(math.log10(args.mazes))
    base_path = args.rules.parent
    for s in set(bd.seed for bd in [bd_0, bd_1]):
        for i in range(args.mazes):
            bd = Maze.BuildData(seed=s)
            for k, v in rules.items():
                v0, v1 = None, None
                try:
                    v0, v1 = getattr(bd_0, k), getattr(bd_1, k)
                    v = eval(v)
                    t0, t1, t = type(v0), type(v1), type(v)
                    if (v0 or v1) and not (t == t0 or t == t1):
                        raise ValueError(f"Incompatible types for field {k}:"
                                         f" {t} != ({t0}, {t1})")
                    setattr(bd, k, v)
                except Exception:
                    raise IOError(
                        f"Invalid interpolation {v0} -> {v1}.")
            maze = Maze.generate(bd)
            mazes.append((i, maze))

    with open(base_path.joinpath("mazes"), 'w') as f:
        f.write("Set ID Name Complexity\n")
        for i, maze in mazes:
            train = int(maze.seed == bd_0.seed)

            name = maze.to_string()
            path = base_path.joinpath(
                f"{train}_{i:0{m_id_digits}d}_{name}.png")
            MazeWidget.draw_to(maze, path, size=256)

            complexity = Simulation.compute_complexity(
                maze, InputType.DISCRETE, 15)['entropy']
            complexity = ' '.join(f"{c:.2}" for c in complexity.values())
            line = f"{train} {i} {maze.to_string()} {complexity}"
            print(line)
            f.write(f"{line}\n")


if __name__ == '__main__':
    main()
