#!/usr/bin/env python3
import os.path
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

import humanize.time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator


def __to_print_string(df, quantiles, header=None):
    def w_max(c):
        return max(*(len(str(v_)) for v_ in df[c]),
                   *(len(str(v_)) for v_ in quantiles[c]),
                   len(c))

    w_idx = max(*(len(str(i)) for i in df.index),
                *(len(str(i)) for i in quantiles.index))

    widths = [w_idx, *(w_max(c) for c in df.columns)]
    formats = [lambda v_, w=w: f"{v_:{w}}" for w in widths]
    df_str = formats[0]('')
    for i, c in enumerate(df.columns):
        df_str += " " + formats[i + 1](c)
    df_str += "\n"
    df_midrule = "-" * len(df_str) + "\n"

    if header:
        df_str = f"{header:^{len(df_midrule)}}\n{df_midrule}{df_str}"

    df_str += df_midrule
    for j, row in quantiles.iterrows():
        df_str += formats[0](j)
        for i, v in enumerate(row):
            df_str += " " + formats[i + 1](v)
        df_str += "\n"
    df_str += df_midrule
    for j, row in df.iterrows():
        df_str += formats[0](j)
        for i, v in enumerate(row):
            df_str += " " + formats[i + 1](v)
        df_str += "\n"
    return df_str


def find_events(root: Path):
    # print(root)
    events = sorted(list(root.glob("**/events.out.*")),
                    key=os.path.getmtime, reverse=True)
    grouped_events = defaultdict(lambda: defaultdict(list))
    for e in events:
        e = e.relative_to(root)
        e_tokens = str(e).split('/')

        grouped_events['/'.join(e_tokens[:-3])][e_tokens[-3]].append(
            (e_tokens[-2], e_tokens[-1]))

    # pprint.pprint(grouped_events)
    fig, axes = plt.subplots(len(grouped_events), 1,
                             sharey='all')

    if len(grouped_events) == 1:
        axes = [axes]

    dfs = {}
    df_strings = []

    # out = defaultdict(lambda: defaultdict(pd.DataFrame))
    for t, groups in grouped_events.items():
        timesteps = {}

        for g, events in groups.items():
            summary_iterators = \
                [EventAccumulator(
                    str(root.joinpath(t).joinpath(g).joinpath(r).joinpath(e))
                ).Reload()
                 for r, e in events]
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
        quantiles = df.quantile([0, .25, .5, .75, 1])

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
            if median >= 1_000_000:
                median_str = f"{median//1_000_000:.0f}M"
            elif median >= 1_000:
                median_str = f"{median//1_000:.0f}K"
            else:
                median_str = str(median)
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

#
# def tabulate_events(root, group, events, out):
#     summary_iterators = \
#         [EventAccumulator(str(root.joinpath(group).joinpath(e))).Reload()
#          for e in events]
#
#     tags = summary_iterators[0].Tags()['scalars']
#     for it in summary_iterators:
#         assert it.Tags()['scalars'] == tags
#
#     # print(f"{group}: {list(i.Scalars(tags[0])[-1].step for i in summary_iterators)}")
#
#     for tag in tags:
#         for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
#             assert len(set(e.step for e in events)) == 1
#             out[tag][group].append([e.value for e in events])
#
#     return out

#
# def write_combined_events(dpath, d_combined, dname='combined'):
#
#     fpath = os.path.join(dpath, dname)
#     writer = tf.summary.FileWriter(fpath)
#
#     tags, values = zip(*d_combined.items())
#
#     timestep_mean = np.array(values).mean(axis=-1)
#
#     for tag, means in zip(tags, timestep_mean):
#         for i, mean in enumerate(means):
#             summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=mean)])
#             writer.add_summary(summary, global_step=i)
#
#         writer.flush()


if __name__ == '__main__':
    start = time.perf_counter()
    find_events(Path(sys.argv[1]))
    print("Computed in",
          humanize.precisedelta(
            timedelta(seconds=time.perf_counter() - start)))
