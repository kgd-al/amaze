#!/usr/bin/env python3

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator


def print_df(df: pd.DataFrame, header: Optional[str] = None):
    if not header:
        print(df)
    else:
        df_str = df.to_string()
        df_str_w = len(df_str.split('\n')[0])
        df_header_deco = "=" * ((df_str_w - len(header) - 2) // 2)
        print(df_header_deco, header, df_header_deco)
        print(df_str)


def find_events(root: Path):
    # print(root)
    events = sorted(list(root.glob("**/events.out.*")))
    grouped_events = defaultdict(lambda: defaultdict(list))
    for e in events:
        e = e.relative_to(root)
        e_tokens = str(e).split('/')

        grouped_events['/'.join(e_tokens[:-3])][e_tokens[-3]].append(
            e_tokens[-2] + "/" + e_tokens[-1])

    # pprint.pprint(grouped_events)
    fig, axes = plt.subplots(len(grouped_events), 1,
                             sharey='all')

    if len(grouped_events) == 1:
        axes = [axes]

    # out = defaultdict(lambda: defaultdict(pd.DataFrame))
    for ax, (t, groups) in zip(axes, grouped_events.items()):
        timesteps = {}

        for g, events in groups.items():
            summary_iterators = \
                [EventAccumulator(str(root.joinpath(t).joinpath(g).joinpath(e))).Reload()
                 for e in events]
            tags = summary_iterators[0].Tags()['scalars']
            timesteps[g] = [i.Scalars(tags[0])[-1].step for i in summary_iterators]
    # pprint.pprint(timesteps)

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in timesteps.items()]))
        print_df(df, "Raw data")
        quantiles = df.quantile([0, .25, .5, .75, 1])
        print_df(quantiles, "Quantiles")
        # print(quantiles.loc[.5])

        sns.violinplot(ax=ax, data=df, cut=0, scale='width')

        ax: Axes = ax
        for i, median in enumerate(quantiles.loc[.5]):
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
                        size='small'
                        )

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
    find_events(Path(sys.argv[1]))
