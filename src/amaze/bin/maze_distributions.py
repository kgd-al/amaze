#!/usr/bin/env python3

import argparse
import concurrent
import itertools
import logging
import os
import pprint
import random
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Optional, List, Sequence

import numpy as np
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
    append: bool = False
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
        parser.add_argument("--append", action="store_true",
                            help="Try to only generate missing entries")


def __bd(clues=False, lures=False, p_lure=0, traps=False, p_trap=0):
    if not clues:
        return Maze.BuildData(unicursive=True)
    return Maze.BuildData(unicursive=False,
                          clue=[Sign()] if clues else [],
                          lure=[Sign()] if lures else [], p_lure=p_lure,
                          trap=[Sign()] if traps else [], p_trap=p_trap)


def _generate(*args, mazes_set, signs, folder):
    seed, size, (class_name, bd), p_lure, p_trap, ssize = args
    clues, lures, traps = signs

    bd.seed = seed + 18
    bd.width = size
    bd.height = size
    bd.p_lure = p_lure
    bd.p_trap = p_trap

    if bd.clue:
        bd.clue = clues[:ssize]
    if bd.lure:
        bd.lure = lures[:ssize]
    if bd.trap:
        bd.trap = traps[:ssize]

    if class_name in ["Trivial", "Simple"] and (p_lure > 0 or p_trap > 0):
        return False

    if class_name in ["Lures", "Complex"] and p_lure == 0:
        return False

    if class_name in ["Traps", "Complex"] and p_trap == 0:
        return False

    maze_str = Maze.bd_to_string(bd)
    if maze_str in mazes_set:
        return False
    else:
        mazes_set.add(maze_str)

    maze = Maze.generate(bd)

    if seed == 0:
        maze_viewer.main(f"--maze {maze_str}"
                         f" --render {folder.joinpath(maze_str)}.png"
                         f" --dark --colorblind"
                         .split(" "))

    stats = maze.stats()
    cm = Simulation.compute_metrics(maze, InputType.DISCRETE, 5)
    stats.update({f"E{k}": v for k, v in cm[MazeMetrics.SURPRISINGNESS].items()})
    stats['Deceptiveness'] = cm[MazeMetrics.DECEPTIVENESS]
    stats['Inseparability'] = cm[MazeMetrics.INSEPARABILITY]
    stats['Name'] = maze_str
    stats['Class'] = class_name
    stats['P_l'] = p_lure
    stats['P_t'] = p_trap
    stats['SSize'] = ssize
    return stats


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
        [0., .25, .5, .75, 1.],
        [0., .25, .5, .75, 1.],
        [1, 2, 3, 4, 5]
    ]

    def _signs(*_args): return [Sign(value=_v) for _v in _args]
    values = np.linspace(1, 0, 15, endpoint=False)
    clues = _signs(*values[0::3])
    lures = _signs(*values[1::3])
    traps = _signs(*values[2::3])
    signs = [clues, lures, traps]

    mazes_set = set()
    if not args.generate and args.append and args.df_path.exists():
        df = pd.read_csv(str(args.df_path), index_col="Name")
        mazes_set = set(df.index.values)
        print(f"Will skip {len(mazes_set)} existing mazes")

    skipped, processed = 0, 0

    def _save_df(*_, interrupt=True):
        if df is not None and args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.df_path)
            if interrupt:
                print(f"Interrupted while generating {args.df_path}")
            else:
                print("Successfully generated", args.df_path)
            print(f"{len(df)} items written ({skipped=}, {processed=})")
        else:
            print("Error generating", args.df_path)
        if interrupt:
            sys.exit()

    signal.signal(signal.SIGINT, _save_df)
    signal.signal(signal.SIGTERM, _save_df)

    # progress_bar = tqdm(desc="Generating",
    #                     total=reduce(mul, [len(lst) for lst in items]))
    # # progress_bar.
    # with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
    #     for g_args in itertools.product(*items):
    #         progress_bar.update()
    #     for future in concurrent.futures.as_completed(
    #         executor.submit(_generate, *g_args,
    #                         mazes_set=mazes_set, signs=signs, folder=args.out)
    #         for g_args in itertools.product(*items)
    #     ):
    #         progress_bar.update()
    #         print(future, future.result())
    #         result = future.result()
    #         if isinstance(result, bool):
    #             skipped += 1
    #         else:
    #             if df is None:
    #                 df = pd.DataFrame(columns=result.keys())
    #             df.loc[len(df)] = result.values()
    # progress_bar.close()

    for g_args in tqdm_iter.product(*items):
        r = _generate(*g_args,
                      mazes_set=mazes_set, signs=signs, folder=args.out)
        if isinstance(r, bool):
            skipped += 1
        else:
            processed += 1
            name = r.pop("Name")
            if df is None:
                df = pd.DataFrame(columns=r.keys(),
                                  index=pd.Index([], name="Name"))
            df.loc[name] = r.values()

    _save_df(interrupt=False)

    return df


def plot(df, args):
    seaborn.set_style("darkgrid")

    df = df.reset_index()
    df["f_size"] = df["size"].apply(lambda s: s.split("x")[0])
    df["Seeds"] = [s.split("_")[0][1:] for s in df["Name"]]

    with (PdfPages(args.pdf_path) as pdf):
        # fig, ax = plt.subplots()
        # df_class = df.groupby("Class")
        # for d_class in ["Complex", "Lures", "Traps", "Simple", "Trivial"]:
        #     ax.scatter(data=df_class.get_group(d_class),
        #                x="Epath", y="Deceptiveness",
        #                label=d_class, s=.25)
        # ax.set_xlabel("Surprisingness")
        # ax.set_ylabel("Deceptiveness")
        # ax.legend()
        # fig.tight_layout()
        # pdf.savefig(fig)

        hue_order = ["Trivial", "Simple", "Traps", "Lures", "Complex"]
        # hue_order = ["Complex", "Lures", "Traps", "Simple", "Trivial"]

        jp_dict = dict(
            y="Deceptiveness",
            marginal_kws=dict(cut=0, common_norm=True),
            joint_kws=dict()
        )

        s = 1
        for e_type, hue_column in tqdm_iter.product(
            ["Epath", "Eall"],
            ["Class", "size", "Seeds", "P_l", "P_t", "SSize"],
            desc="Joint plots"
        ):
            g = seaborn.jointplot(data=df, x=e_type, hue=hue_column,
                                  palette="deep",
                                  s=s, linewidths=0,
                                  **jp_dict)
            g.plot_joint(seaborn.kdeplot, zorder=0, fill=True, alpha=.5,
                         warn_singular=False)
            pdf.savefig(g.figure)

        # mini_df = df[["Class", "size", "Prob.", "SSize", "Epath", "Eall"]].reset_index()
        # mini_df = pd.wide_to_long(mini_df,
        #                           "E", i="index", j="Entropy", suffix=r'\D+'
        #                           ).reset_index()
        # print(mini_df)
        # for col in tqdm(["Prob.", "SSize"], desc="Violin plots (split)"):
        #     g = seaborn.catplot(data=mini_df, x="Class", hue="Entropy", y="E",
        #                         col=col, row='size', split=True, dodge=True,
        #                         density_norm="width",
        #                         kind='violin', cut=0, width=.8,
        #                         palette="pastel", inner='quart')
        #     g.figure.tight_layout()
        #
        #     pdf.savefig(g.figure)
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
    if args.generate or args.append or not args.df_path.exists():
        df = generate(args)
    else:
        df = pd.read_csv(args.df_path, index_col="Name")

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
