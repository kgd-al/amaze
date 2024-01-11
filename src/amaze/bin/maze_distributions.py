#!/usr/bin/env python3

import argparse
import pprint
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Sequence

import pandas as pd

from amaze.simu.env.maze import Maze
from amaze.simu.robot import InputType
from amaze.simu.simulation import Simulation


@dataclass
class Options:
    n: int = 1_000
    out: Path = Path("tmp/maze_distributions")
    df_path = "distributions.csv"

    generate: bool = True
    plot: bool = True

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("-n",
                            help="How many mazes to sample from")
        parser.add_argument("--out", type=Path,
                            help="Where to store the results")

        parser.add_argument("--no-plot", action="store_false", dest="plot",
                            help="Do not generate summary plots")
        parser.add_argument("--no-generate", action="store_false", dest="generate",
                            help="Use existing sampled data (if any)")


def generate(args):
    bd = Maze.BuildData(width=10, height=10, unicursive=True,
                        clue=[],
                        lure=[], p_lure=0,
                        trap=[], p_trap=0)
    df = None
    for i in range(args.n):
        bd.seed = i
        maze = Maze.generate(bd)
        stats = maze.stats()
        c = Simulation.compute_complexity(maze, InputType.DISCRETE, 5)['entropy']
        stats.update({f"E{k}": v for k, v in c.items()})
        stats['Name'] = maze.to_string()
        if df is None:
            df = pd.DataFrame(columns=stats.keys())
        df.loc[len(df)] = stats.values()
    print(df)

    if df is not None and args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.df_path)

    return df


def plot(df, args):
    pass


def main(sys_args: Optional[Sequence[str]] = None):
    args = Options()
    parser = argparse.ArgumentParser(
        description="Generates distributions of mazes statistics")
    Options.populate(parser)
    parser.parse_args(args=sys_args, namespace=args)

    args.df_path = args.out.joinpath(args.df_path)
    if args.generate or not args.df_path.exists():
        df = generate(args)
    else:
        df = pd.read_csv(args.df_path)

    if args.plot:
        plot(df, args)

    return 0


if __name__ == '__main__':
    exit(main())
