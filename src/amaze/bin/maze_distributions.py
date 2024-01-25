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
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LinearLocator
from tqdm import tqdm
import tqdm.contrib.itertools as tqdm_iter

from amaze.bin import maze_viewer
from amaze.simu._maze_metrics import MazeMetrics
from amaze.simu.env.maze import Maze
from amaze.simu.robot import InputType
from amaze.simu.simulation import Simulation
from amaze.visu.resources import Sign
from amaze.visu.widgets.maze import MazeWidget


@dataclass
class Options:
    n: int = 1_000
    out: Path = Path("tmp/maze_distributions")
    df_path, pdf_path = None, None

    generate: bool = False
    plot: bool = True

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("-n", type=int,
                            help="How many mazes to sample from")
        parser.add_argument("--out", type=Path,
                            help="Where to store the results")

        parser.add_argument("--no-plot", action="store_false", dest="plot",
                            help="Do not generate summary plots")
        parser.add_argument("--force-generate", action="store_true", dest="generate",
                            help="Force sampled data regeneration (if any)")


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

    mazes_set = set()

    for seed, size, (class_name, bd), v in (
            tqdm_iter.product(*items, desc="Processing")):
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

        if class_name in ["Trivial", "Simple"] and p > 0:
            continue

        maze_str = Maze.bd_to_string(bd)
        if maze_str in mazes_set:
            continue
        else:
            mazes_set.add(maze_str)

        maze = Maze.generate(bd)

        if seed == 0:
            maze_viewer.main(f"--maze {maze_str}"
                             f" --render {args.out.joinpath(maze_str)}.png"
                             f" --dark --colorblind"
                             .split(" "))

        stats = maze.stats()
        cm = Simulation.compute_metrics(maze, InputType.DISCRETE, 5)
        stats.update({f"E{k}": v for k, v in cm[MazeMetrics.SURPRISINGNESS].items()})
        stats['Deceptiveness'] = cm[MazeMetrics.DECEPTIVENESS]
        stats['Inseparability'] = cm[MazeMetrics.INSEPARABILITY]
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
    seaborn.set_style("darkgrid")

    df["f_size"] = df["size"].apply(lambda s: s.split("x")[0])

    with (PdfPages(args.pdf_path) as pdf):
        fig, ax = plt.subplots()
        df_class = df.groupby("Class")
        for d_class in ["Complex", "Lures", "Traps", "Simple", "Trivial"]:
            ax.scatter(data=df_class.get_group(d_class),
                       x="Epath", y="Deceptiveness",
                       label=d_class, s=.25)
        ax.set_xlabel("Surprisingness")
        ax.set_ylabel("Deceptiveness")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)

        mini_df = df[["Class", "size", "Prob.", "SSize", "Epath", "Eall"]].reset_index()
        mini_df = pd.wide_to_long(mini_df,
                                  "E", i="index", j="Entropy", suffix=r'\D+'
                                  ).reset_index()
        for col in tqdm(["Prob.", "SSize"], desc="Violin plots (dodge)"):
            g = seaborn.catplot(data=mini_df, x="Class", hue="Entropy", y="E",
                                col=col, row='size', dodge=True, split=True,
                                density_norm="width",
                                kind='violin', cut=0, width=.8,
                                palette="pastel", inner='quart')
            g.figure.tight_layout()

            pdf.savefig(g.figure)
        #
        # for col, y in tqdm_iter.product(["Prob.", "SSize"],
        #                                 ["Deceptiveness", "Inseparability"],
        #                                 desc="Violin plots (straight)"):
        #     g = seaborn.catplot(data=df, x="Class", hue="Class", y=y,
        #                         col=col, row='size',
        #                         density_norm="width",
        #                         kind='violin', cut=0, width=.8,
        #                         palette="pastel", inner='quart')
        #     g.figure.tight_layout()
        #
        #     pdf.savefig(g.figure)

        # for col, x, y in tqdm_iter.product(["Prob.", "SSize"],
        #                                    ["path", "intersections"],
        #                                    ["Epath", "Eall"],
        #                                    desc="Strip plots"):
        #     g = seaborn.catplot(data=df, x=x, y=y, hue="Class", col=col,
        #                         row="size", kind='strip')
        #     for ax in g.axes[-1]:
        #         ax.xaxis.set_major_locator(LinearLocator(5))
        #     g.figure.tight_layout()
        #     pdf.savefig(g.figure)
    print("Generated", args.pdf_path)


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

    args.pdf_path = args.df_path.with_suffix(".pdf")
    if args.plot:
        plot(df, args)

    pdf_symlink = args.out.joinpath("distributions.pdf")
    pdf_symlink.unlink(missing_ok=True)
    pdf_symlink.symlink_to(args.pdf_path.name)
    print("Generated", pdf_symlink, "->", args.pdf_path)

    return 0


if __name__ == '__main__':
    exit(main())
