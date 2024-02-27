#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Sequence

import pandas as pd

from amaze.simu.maze import Maze
from amaze.simu.types import InputType
from amaze.simu.simulation import Simulation


@dataclass
class Options:
    mazes: List[str] = field(default_factory=list)
    file: Optional[Path] = None
    out: Optional[Path] = None

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("mazes", metavar="maze",
                            nargs='*',
                            help="Extract stats from provided maze")
        parser.add_argument("--file",
                            help="Read provided panda dataframe and outputs"
                                 " with added stats columns")
        parser.add_argument("--out",
                            help="Store formatted table into target file")


def _get_stats(maze: str):
    m = Maze.from_string(maze)
    stats = m.stats()
    c = Simulation.compute_complexity(m, InputType.DISCRETE, 5)['entropy']
    stats.update({f"E{k}": v for k, v in c.items()})
    return stats


def main(sys_args: Optional[Sequence[str]] = None):
    """ Tool to print stats about mazes without having to simulate them """
    args = Options()
    parser = argparse.ArgumentParser(
        description="Outputs stats for selected maze in a multitude of formats")
    Options.populate(parser)
    parser.parse_args(args=sys_args, namespace=args)

    if args.file:
        in_df = pd.read_csv(args.file, sep=None, engine='python',
                            index_col=False)
        stats_df = None
        try:
            name_index = list(in_df.columns).index("Name")
        except ValueError:
            raise KeyError("No name column in provided file")
        in_df.set_index("Name", inplace=True)
        for i, r in in_df.iterrows():
            # print(f"{i=}, {r=}")
            stats = _get_stats(str(i))
            if stats_df is None:
                stats_df = pd.DataFrame(columns=stats.keys())
            stats_df.loc[i] = stats.values()
        df = in_df.join(stats_df)
        print(df)

    else:
        df = None
        for maze_str in args.mazes:
            maze = Maze.from_string(maze_str)
            stats = maze.stats()
            stats['Name'] = maze_str
            if df is None:
                df = pd.DataFrame(columns=stats.keys())
            df.loc[len(df)] = stats.values()
        print(df)

    if df is not None and args.out:
        df.to_csv(args.out)

    return 0


if __name__ == '__main__':
    exit(main())
