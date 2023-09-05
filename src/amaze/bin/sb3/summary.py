#!/usr/bin/env python3
import os.path
import pprint
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

import humanize.time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QApplication
from matplotlib.axes import Axes
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

from amaze.simu.env.maze import Maze
from amaze.simu.robot import InputType
from amaze.simu.simulation import Simulation
from amaze.visu import resources
from amaze.visu.widgets.labels import InputsLabel
from amaze.visu.widgets.maze import MazeWidget


def __to_km_string(v):
    if v >= 1_000_000:
        return f"{v // 1_000_000:.0f}M"
    elif v >= 1_000:
        return f"{v // 1_000:.0f}K"
    else:
        return str(v)


def __to_print_string(df, quantiles, header=None):
    min_v = np.min(quantiles.loc[0.0])

    if min_v >= 1_000_000:
        scale = 1_000_000
    elif min_v >= 1_000:
        scale = 1_000
    else:
        scale = 1

    def w_max(c):
        return max(*(len(str(v_ // scale)) for v_ in df[c]),
                   *(len(str(v_ // scale)) for v_ in quantiles[c]),
                   len(c))

    w_idx = max(*(len(str(i)) for i in df.index),
                *(len(str(i)) for i in quantiles.index))

    widths = [w_idx, *(w_max(c) for c in df.columns)]
    formats = [lambda v_, w=w: f"{v_:{w}}" for w in widths]
    df_str = formats[0]('')
    for i, c in enumerate(df.columns):
        df_str += " " + f"{c:^{widths[i+1]}}"
    df_str += "\n"
    df_midrule = "-" * len(df_str) + "\n"

    if scale > 1:
        if header:
            header += " "
        header += f"(x{scale})"
    if header:
        df_str = f"{header:^{len(df_midrule)}}\n{df_midrule}{df_str}"

    df_str += df_midrule
    for j, row in quantiles.iterrows():
        df_str += formats[0](j)
        for i, v in enumerate(row):
            df_str += " " + formats[i + 1](v // scale)
        df_str += "\n"
    df_str += df_midrule
    for j, row in df.iterrows():
        df_str += formats[0](j)
        for i, v in enumerate(row):
            df_str += " " + formats[i + 1](v // scale)
        df_str += "\n"
    return df_str


def find_events(root: Path):
    # print(root)
    events = sorted(list(root.glob("**/events.out.*")),
                    key=os.path.getmtime, reverse=True)
    grouped_events = defaultdict(lambda: defaultdict(list))
    for e in events:
        e_ = e.relative_to(root)
        e_tokens = str(e_).split('/')
        group = '/'.join(e_tokens[:-3])
        maze, run, file = e_tokens[-3:]

        if group:
            grouped_events[group][maze].append((run, e))
        else:
            grouped_events[maze][run[-2]].append((run[-1], e))

    # pprint.pprint(grouped_events)

    n_rows = len(grouped_events)
    fig, axes = plt.subplots(n_rows, 1,
                             sharey='all',
                             figsize=(10, n_rows * 2))

    if len(grouped_events) == 1:
        axes = [axes]

    dfs = {}
    df_strings = []

    # out = defaultdict(lambda: defaultdict(pd.DataFrame))
    for t, groups in grouped_events.items():
        timesteps = {}

        for g, events in groups.items():
            summary_iterators = \
                [EventAccumulator(str(e)).Reload() for r, e in events]
            tags = summary_iterators[0].Tags()['scalars']
            timesteps[g] = (
                [r for r, _ in events],
                [i.Scalars(tags[0])[-1].step for i in summary_iterators]
            )
        # pprint.pprint(timesteps)

        time_order = []
        for runs, _ in timesteps.values():
            for r in runs:
                if r not in time_order:
                    time_order.append(r)

        df = pd.DataFrame(dict([(k, pd.Series(v, index=i))
                                for k, (i, v) in timesteps.items()]),
                          index=time_order)
        quantiles = df.quantile([0, .25, .5, .75, 1]).astype(df.dtypes)

        # print(df)
        # print(quantiles)
        df_strings.append(__to_print_string(df, quantiles, t))
        dfs[t] = (df, quantiles)

    for ax, (t, (df, q)) in zip(axes, sorted(dfs.items())):
        # Restore lexicographic order
        df = df.reindex(sorted(df.columns), axis=1)
        q = q.reindex(sorted(q.columns), axis=1)

        # Plot
        sns.violinplot(ax=ax, data=df, cut=0, scale='width')

        ax: Axes = ax
        for i, median in enumerate(q.loc[.5]):
            x = i+.1
            median_str = __to_km_string(median)
            # ax.plot((i, x), (median, median), color='gray')
            # ax.text(x, median, median_str,
            #         )
            ax.annotate(text=median_str,
                        xy=(i, median),
                        xytext=(x, median),
                        horizontalalignment='center',
                        verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.5),
                        arrowprops=dict(arrowstyle='-', linestyle='dotted'),
                        size='small')

        ax.set(yscale="log")
        if ax == axes[-1]:
            ax.set_xlabel('Maze class')
        else:
            ax.set_xticks([])
        ax.set_ylabel('Timesteps')
        ax.grid(axis='y')

        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(t)

    fig.savefig(root.joinpath("timesteps.png"), bbox_inches='tight')

    max_w = max(*[len(line) for s in df_strings for line in s.split("\n")])
    for elems in zip(*[s.split("\n") for s in df_strings]):
        print("\t".join(f"{e:{max_w}}" for e in elems))


def process(training_type, folders, df):
    if df is None:
        df = pd.DataFrame(columns=["Trainer", "Algo", "Replicate",
                                   "Time", "Reward"])
    for f in folders:
        paths = None
        if tt == 'direct':
            paths = f.glob("*events*")

        elif tt == 'interpolation':
            paths = f.glob("*events*")

        elif tt == 'edhucat':
            paths = f.glob("timeline/*events*")

        last = sorted(list(paths), key=os.path.getmtime, reverse=False)[-1]
        tokens = str(last).split('/')
        algo = tokens[1].split('-')[1]
        e = (EventAccumulator(str(last)).Reload()
             .Scalars('infos/pretty_reward'))[-1]
        df = pd.concat([
            pd.DataFrame([[tt, algo, tokens[2], e.step, e.value]],
                         columns=df.columns),
            df])
    return df


if __name__ == '__main__':
    start = time.perf_counter()

    # training_types = ["edhucat"]
    training_types = ["direct", "interpolation", "edhucat"]

    out_folder = Path("results/train_summary/")
    dataframe_file = out_folder.joinpath("stats.csv")
    if False and dataframe_file.exists():
        dataframe = pd.read_csv(dataframe_file)
    else:
        dataframe = None
        for tt in training_types:
            dataframe = (
                process(tt, sorted(list(Path(".").glob(f"results/{tt}*/**/run*"))),
                        dataframe))
        print(dataframe)
        dataframe.to_csv(dataframe_file, index=False)

    common_args = dict(x="Trainer", y="Reward", hue="Algo",
                       order=training_types,)
    g = seaborn.catplot(data=dataframe, **common_args,
                        kind='swarm',
                        aspect=5)
    g.map_dataframe(seaborn.violinplot, **common_args,
                    split=True, cut=0, scale='width',
                    palette='pastel')

    seaborn.move_legend(g, "upper center", ncol=3, title=None)
    g.figure.tight_layout()
    g.figure.savefig(out_folder.joinpath('stats.pdf'))

    # find_events(Path(sys.argv[1]))

    print("Computed in",
          humanize.precisedelta(
            timedelta(seconds=time.perf_counter() - start)))
