#!/usr/bin/env python3

import argparse
import itertools
import logging
import pprint
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Optional, List, Sequence

import pandas as pd
import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from amaze.bin import maze_viewer
from amaze.simu.env.maze import Maze
from amaze.simu.robot import InputType
from amaze.simu.simulation import Simulation
from amaze.visu.resources import Sign
from amaze.visu.widgets.maze import MazeWidget


@dataclass
class Options:
    n: int = 1_000
    out: Path = Path("tmp/maze_distributions")
    df_path = None

    generate: bool = True
    plot: bool = True

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("-n", type=int,
                            help="How many mazes to sample from")
        parser.add_argument("--out", type=Path,
                            help="Where to store the results")

        parser.add_argument("--no-plot", action="store_false", dest="plot",
                            help="Do not generate summary plots")
        parser.add_argument("--no-generate", action="store_false", dest="generate",
                            help="Use existing sampled data (if any)")


def __bd(clues=False, lures=False, p_lure=0, traps=False, p_trap=0):
    if not clues:
        return Maze.BuildData(unicursive=True)
    return Maze.BuildData(unicursive=False,
                          clue=[Sign()] if clues else [],
                          lure=[Sign()] if lures else [], p_lure=p_lure,
                          trap=[Sign()] if traps else [], p_trap=p_trap)


def generate(args):
    df = None
    items = [
        range(args.n),      # Seeds
        [5, 10, 25, 50],    # Sizes
        [                   # Classes
            ("Trivial", __bd()),
            ("Simple", __bd(clues=True)),
            ("Traps", __bd(clues=True, traps=True)),
            ("Lures", __bd(clues=True, lures=True)),
            ("Complex", __bd(clues=True, lures=True, traps=True)),
        ],
        [0., .25, .5, .75, 1.] + [1, 2, 3, 4, 5]
    ]

    def signs(*_args): return [Sign(value=_v) for _v in _args]
    clues = signs(.9, 0.85, 0.8, 0.75, 0.7)
    lures = signs(0.6, 0.55, 0.5, 0.45, 0.4)
    traps = signs(0.3, 0.25, 0.2, 0.15, 0.1)

    for seed, size, (class_name, bd), v in (
            tqdm(itertools.product(*items), desc="Processing",
                 total=reduce(mul, [len(lst) for lst in items]))):
        if isinstance(v, int):
            ssize = v
            p = .5
        else:
            ssize = 1
            p = v

        bd.seed = seed + 10
        bd.width = size
        bd.height = size
        bd.p_lure = p
        bd.p_trap = p

        if bd.clue:
            bd.clue = clues[:ssize]
        if bd.lure:
            bd.lure = lures[:ssize]
        if bd.trap:
            bd.trap = traps[:ssize]

        maze = Maze.generate(bd)
        maze_str = maze.to_string()

        if seed == 0:
            maze_viewer.main(f"--maze {maze_str}"
                             f" --render {args.out.joinpath(maze_str)}.png"
                             f" --dark --colorblind"
                             .split(" "))

        stats = maze.stats()
        c = Simulation.compute_complexity(maze, InputType.DISCRETE, 5)['entropy']
        stats.update({f"E{k}": v for k, v in c.items()})
        stats['Name'] = maze_str
        stats['Class'] = class_name
        stats['Prob.'] = p
        stats['SSize'] = ssize
        if df is None:
            df = pd.DataFrame(columns=stats.keys())
        df.loc[len(df)] = stats.values()
    print(df.columns)
    print(df)

    if df is not None and args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.df_path)
        print("Generated", args.df_path)

    return df


def plot(df, args):
    out = args.df_path.with_suffix(".pdf")
    seaborn.set_style("darkgrid")
    with PdfPages(out) as pdf:
        for row in ["Prob.", "SSize"]:
            for c in ["Epath", "Eall"]:
                common_args = dict(x="Class", y=c, col=row, row='size')
                violin_args = dict(
                    kind='violin', cut=0, scale='count', palette='pastel'
                )
                g = seaborn.catplot(data=df, **common_args, **violin_args)
                g.figure.tight_layout()

                pdf.savefig(g.figure)
    print("Generated", out)


def main(sys_args: Optional[Sequence[str]] = None):
    args = Options()
    parser = argparse.ArgumentParser(
        description="Generates distributions of mazes statistics")
    Options.populate(parser)
    parser.parse_args(args=sys_args, namespace=args)

    logging.basicConfig(level=logging.WARNING, force=True)

    args.df_path = args.out.joinpath(f"distributions_{args.n}.csv")
    if args.generate or not args.df_path.exists():
        df = generate(args)
    else:
        df = pd.read_csv(args.df_path)

    if args.plot:
        plot(df, args)

    return 0


if __name__ == '__main__':
    exit(main())
