#!/usr/bin/env python3

import argparse
import concurrent
import itertools
import logging
import numbers
import os
import pprint
import random
import signal
import sys
from collections import Counter
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

    parallel: int = 1

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
        parser.add_argument("--parallel", type=int,
                            help="Use multiple process")


def __bd(clues=False, lures=False, p_lure=0, traps=False, p_trap=0):
    if not clues:
        return Maze.BuildData(unicursive=True)
    return Maze.BuildData(unicursive=False,
                          clue=[Sign()] if clues else [],
                          lure=[Sign()] if lures else [], p_lure=p_lure,
                          trap=[Sign()] if traps else [], p_trap=p_trap)


def _signs(*_args): return [Sign(value=_v) for _v in _args]


values = np.linspace(1, 0, 15, endpoint=False)
clues = _signs(*values[0::3])
lures = _signs(*values[1::3])
traps = _signs(*values[2::3])


def _generate(*args, mazes_set):
    seed, size, (class_name, bd), p_lure, p_trap, ssize = args
    # print(f"[Start] _generate({args})")

    bd.seed = seed
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

    if class_name != "Trivial" and maze.unicursive():
        return False

    if class_name in ["Lures", "Complex"] and len(maze.lures()) == 0:
        return False

    if class_name in ["Traps", "Complex"] and len(maze.traps()) == 0:
        return False

    stats = maze.stats()
    cm = Simulation.compute_metrics(maze, InputType.DISCRETE, 5)
    stats.update({f"E{k}": v for k, v in cm[MazeMetrics.SURPRISINGNESS].items()})
    # stats.update(cm[MazeMetrics.DECEPTIVENESS])
    stats['Deceptiveness'] = cm[MazeMetrics.DECEPTIVENESS]
    stats['Inseparability'] = cm[MazeMetrics.INSEPARABILITY]
    stats['Name'] = maze_str
    stats['Class'] = class_name
    stats['P_l'] = p_lure
    stats['P_t'] = p_trap
    stats['SSize'] = ssize

    # print(f"[Done] _generate({args})")

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
        [0., .25, .5, .75, 1.],  # Lures probability
        [0., .25, .5, .75, 1.],  # Traps probability
        [1, 2, 3, 4, 5]  # Sets size
    ]

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

    def _add_to_df(_result):
        nonlocal skipped, processed, df
        if isinstance(r, bool):
            skipped += 1
        else:
            processed += 1
            name = r.pop("Name")
            if df is None:
                df = pd.DataFrame(columns=r.keys(),
                                  index=pd.Index([], name="Name"))
            df.loc[name] = r.values()

    if args.parallel > 1:
        progress_bar = tqdm(desc="Generating",
                            total=reduce(mul, [len(lst) for lst in items]))
        # progress_bar.
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            for future in concurrent.futures.as_completed(
                    executor.submit(_generate, *g_args)
                    for g_args in itertools.product(*items)):
                progress_bar.update()
                result = future.result()
                print(future, result, flush=True)
                _add_to_df(result)
        progress_bar.close()

    else:
        for g_args in tqdm_iter.product(*items):
            r = _generate(*g_args, mazes_set=mazes_set)
            _add_to_df(r)

                # if seed == 0:
                #     maze_viewer.main(f"--maze {maze_str}"
                #                      f" --render {folder.joinpath(maze_str)}.png"
                #                      f" --dark --colorblind"
                #                      .split(" "))

    _save_df(interrupt=False)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    return df


def plot(df, args):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    seaborn.set_style("whitegrid")

    df = df.reset_index()
    df["f_size"] = df["size"].apply(lambda s: s.split("x")[0])
    df["Seeds"] = [s.split("_")[0][1:] for s in df["Name"]]

    pretty_labels = {
        "Epath": "Surprisingness", "Eall": "Surprisingness (all inputs)",

        "Class": "Maze class",
        "size": "Maze size", "Seeds": "RNG seed",
        "P_l": "Lures prob.", "P_t": "Traps prob.",
        "SSize": "Unique signs",

        "path": "Optimal path length",
        "intersections": "Number of intersections",
        "clues": "Number of clues",
        "lures": "Number of lures",
        "traps": "Number of traps",
    }
    common_norm = True
    jp_dict = dict(
        kind='kde',
        palette="colorblind",
        joint_kws=dict(cut=0, alpha=1, fill=True,
                       common_norm=common_norm, warn_singular=False),
        marginal_kws=dict(cut=0, common_norm=common_norm,
                          linewidth=.5,
                          warn_singular=False),
    )
    sp_dict = dict(zorder=1, linewidth=0, s=1, legend=False)
    extend = .01
    sample_size = 1000

    if len(df) > sample_size:
        def _get(d_class, col):
            return df.loc[[
                df[df.Class == d_class][col].idxmin(),
                df[df.Class == d_class][col].idxmax()
            ]]

        df_sample = pd.concat(
            [_get(_c, "Epath") for _c in ["Trivial", "Simple"]]
            + [
                _get(_c, _v) for _c, _v in itertools.product(
                    ["Traps", "Lures", "Complex"], ["Epath", "Deceptiveness"])
            ]
        )
        df_sample.to_csv(args.df_path.with_suffix(".extremum.csv"))
        df_sample = pd.concat([
            df_sample.iloc[::-1],
            df.sample(sample_size - len(df_sample), random_state=0)
        ])
        df_sample.to_csv(args.df_path.with_suffix('.sample.csv'))
    else:
        df_sample = df

    hue_orders = dict(Class=["Complex", "Lures", "Traps", "Simple", "Trivial"],
                      size=["50x50", "25x25", "10x10", "5x5"],
                      P_l=[1, .75, .5, .25, 0],
                      P_t=[1, .75, .5, .25, 0],
                      SSize=[5, 4, 3, 2, 1])

    def __plot_one(_pdf, x, y, hue, title=None):
        hue_order = hue_orders.get(hue, [
            k[0] for k in sorted(Counter(df[hue]).items(),
                                 key=lambda _x: _x[1],
                                 reverse=True)])
        g = seaborn.jointplot(data=df, x=x, y=y, hue=hue, **jp_dict,
                              hue_order=hue_order)

        seaborn.scatterplot(data=df_sample,
                            x=x, y=y, hue=hue, ax=g.ax_joint,
                            hue_order=hue_order, palette=jp_dict["palette"],
                            **sp_dict)

        for i, c in enumerate(g.ax_marg_x.collections):
            c.set_zorder(100-i)
        for i, c in enumerate(g.ax_marg_y.collections):
            c.set_zorder(100-i)

        seaborn.despine(bottom=True)

        if "cut" in jp_dict["joint_kws"]:
            x_min, x_max = g.ax_joint.get_xlim()
            x_range = x_max - x_min
            g.ax_joint.set_xlim(x_min - extend * x_range,
                                x_max + extend * x_range)

            y_min, y_max = g.ax_joint.get_ylim()
            y_range = y_max - y_min
            g.ax_joint.set_ylim(y_min - extend * y_range,
                                y_max + extend * y_range)

        g.ax_joint.set_xlabel(pretty_labels.get(x, x))
        g.ax_joint.set_ylabel(pretty_labels.get(y, y))
        g.ax_joint.legend_.set_title(pretty_labels.get(hue, hue))

        if title:
            g.figure.suptitle(title)
            g.figure.tight_layout()

        _pdf.savefig(g.figure, bbox_inches='tight')
        plt.close(g.figure)

    with PdfPages(args.pdf_path.with_suffix('.best.pdf')) as pdf:
        jp_dict["height"] = 3.5
        with seaborn.plotting_context(rc={'legend.fontsize': 'x-small',
                                          'legend.title_fontsize': 'x-small'}):
            __plot_one(pdf, "Epath", "Deceptiveness", "Class")
        del jp_dict["height"]
        # exit(1)

    with (PdfPages(args.pdf_path) as pdf):

        def _plot_one(x, y, hue, title=None):
            __plot_one(pdf, x, y, hue, title)

        def _section(title, body):
            fig, ax = plt.subplots()
            fig.suptitle(title)
            ax.text(.5, .5, body,
                    horizontalalignment='center', verticalalignment='center')
            ax.set_axis_off()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        _section("General stats",
                 f"{len(df)} unique mazes")

        numeric_columns = [c for c in df.columns
                           if pd.api.types.is_numeric_dtype(df[c])]
        _section("Distributions",
                 f"{len(numeric_columns)} variables:\n"
                 + "\n".join(f"- {pretty_labels.get(c, c)}"
                             for c in numeric_columns))
        for c in tqdm(numeric_columns, desc="histograms"):
            g = seaborn.histplot(data=df, x=c)
            g.set_xlabel(pretty_labels.get(c, c))
            pdf.savefig(g.figure)
            plt.close(g.figure)

        _section("Systematic distributions",
                 "Both Surprisingness implementations plotted against"
                 " Deceptiveness.\n"
                 "Group variables are the maze class, size, probabilities and"
                 "number of signs")

        d_variables = ["Deceptiveness"]
        e_variables = ["Epath", "Eall"]
        hue_variables = ["Class", "size", "P_l", "P_t", "SSize"]
        if len(set(df["Seeds"].values)) <= 10:
            hue_variables.append("Seeds")
        for d_type, e_type, hue_column in tqdm_iter.product(
            d_variables, e_variables, hue_variables,
            desc="Joint plots"
        ):
            _plot_one(e_type, d_type, hue_column)

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
