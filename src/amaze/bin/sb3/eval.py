#!/usr/bin/env python3
import argparse
import glob
import itertools
import math
import pprint
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QImage, QPainter, QColor
from PyQt5.QtWidgets import QApplication
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.text import Annotation
from scipy.stats import mannwhitneyu, ttest_ind, false_discovery_control
from statannotations.Annotator import Annotator
from tqdm import tqdm

from amaze.bin import maze_viewer
from amaze.bin.sb3.common import (move_legend, set_seaborn_style, X_ORDER,
                                  HUE_ORDER, SWARM_ARGS, VIOLIN_ARGS, FIG_SIZE_INCHES)
from amaze.sb3.callbacks import recurse_avg_dict
from amaze.sb3.utils import CV2QTGuard
from amaze.simu.controllers.control import load
from amaze.simu.env.maze import Maze
from amaze.simu.pos import Vec
from amaze.simu.robot import InputType, Robot, OutputType, Action
from amaze.simu.simulation import Simulation
from amaze.visu import resources
from amaze.visu.resources import SignType
from amaze.visu.widgets.maze import MazeWidget


def _pretty_delta(start=None):
    if start is None:
        return time.perf_counter()
    else:
        return humanize.precisedelta(
            timedelta(seconds=_pretty_delta() - start))


def _generate_mazes(file: Path):
    start = _pretty_delta()

    levels = [("U", "Trivial"),
              ("C1", "Simple"),
              ("C1_l.25_L.25", "Lures"),
              ("C1_l.25_L.25_t.5_T.5", "Traps")]
    seeds = 10_000
    candidates = len(levels) * seeds

    def make_maze(lvl, seed):
        return f"M{seed + 1_000_000}_20x20_{lvl}"

    maze_header = None

    df = pd.DataFrame()
    traps = pd.Series()
    for (level, name), j in tqdm(itertools.product(levels, range(seeds)),
                                 total=candidates, desc="Generating mazes"):
        maze_str = make_maze(level, j)
        maze = Maze.from_string(maze_str)
        complexity = Simulation.compute_complexity(maze,
                                                   InputType.DISCRETE, 15)

        if maze_header is None:
            maze_header = [k[0].upper() + k[1:].lower()
                           for k in maze.stats().keys()]

        key = name
        if maze.p_trap:
            traps.loc[j] = len(maze.signs_data[SignType.TRAP])

        df.loc[j, key] = complexity['entropy']['min']

    folder = file.parent
    folder.mkdir(parents=True, exist_ok=True)
    fig, _ = plt.subplots()
    seaborn.violinplot(df, cut=0)
    fig.savefig(folder.joinpath('complexity.png'))
    fig, _ = plt.subplots()
    seaborn.violinplot(traps, cut=0)
    fig.savefig(folder.joinpath('traps.png'))

    def arg_quantiles(data):
        quantiles = np.quantile(data, [0, .5, 1])
        return [(np.abs(data - v)).argmin() for v in quantiles]

    mazes = []
    for i, ((level, name), indices) \
            in enumerate(zip(levels,
                             [arg_quantiles(df[c]) for c in df.columns[:-1]])):
        for k, j in enumerate(indices):
            mazes.append(((i, k, j), name, make_maze(level, j)))

    traps_ms = defaultdict(list)
    for i, t in enumerate(traps):
        traps_ms[t].append(i)

    traps_filtered_ms = {t: v for t, v in traps_ms.items()
                         if len(v) >= 3 and t > 0}
    traps_filtered = pd.DataFrame(columns=["Traps", "Complexity"])
    for t, v in traps_filtered_ms.items():
        for j in v:
            traps_filtered.loc[j] = (t, df.loc[j, "Traps"])

    for it, ft in enumerate(np.quantile(traps_filtered, [0, .5, 1])):
        t = round(ft)
        t_mazes = traps_filtered[traps_filtered["Traps"] == t]
        for k, j in enumerate(arg_quantiles(t_mazes["Complexity"])):
            j = t_mazes.index[j]
            mazes.append(((3 + it, k, j), f"Traps_{t}",
                          make_maze(levels[-1][0], j)))

    for i, (_, name) in enumerate(levels):
        folder.joinpath(f"{i}_{name}").mkdir(parents=True, exist_ok=True)

    mdf = pd.DataFrame(columns=["Complexity", *maze_header],
                       index=pd.MultiIndex(
                           levels=[[], [], [], []], codes=[[], [], [], []],
                           names=["Type", "TI", "Name", "MI"]))
    for (i, k, j), t, m in mazes:
        maze = Maze.from_string(m)

        mdf.loc[(t, i, m, k), :] = (df[t.split('_')[0]][j],
                                    *maze.stats().values())
        maze_viewer.main(["--maze", m, "--render",
                          str(folder.joinpath(f"{i}_{t}/{k}__{m}.png"))])
    mdf.to_csv(file)

    print(f"Generated {len(mazes)} mazes from {candidates} candidates"
          f" in {_pretty_delta(start)}")


def merge_images(images: list[QImage]):
    assert len(set(img.width() for img in images))
    assert len(set(img.height() for img in images))
    w, h = images[0].width(), images[0].height()
    big_img = QImage(2 * w, 2 * h, QImage.Format_RGB32)
    big_img.fill(Qt.white)
    painter = QPainter(big_img)
    for ix, img in enumerate(images):
        i, j = ix % 2, ix // 2
        painter.drawImage(i * w, j * h, img)
    painter.end()
    return big_img


ARROW_PATH = resources.arrow_path()
DEBUG_IMAGE_SIZE = 32


def __debug_draw_input(array: np.ndarray, correct_action: Action,
                       selected_action: Action = None):
    w = .1
    image = QImage(DEBUG_IMAGE_SIZE, DEBUG_IMAGE_SIZE, QImage.Format_RGB32)
    image.fill(Qt.black)
    painter = QPainter(image)
    painter.setPen(Qt.white)
    painter.scale(DEBUG_IMAGE_SIZE, DEBUG_IMAGE_SIZE)
    painter.translate(.5, .5)

    for i, v in enumerate(array[:4]):
        if v == 1:
            painter.fillRect(QRectF(.5 - w, -.5, w, 1),
                             QColor.fromHsvF(0, 0, v))
        elif v == .5:
            painter.fillRect(QRectF(.5 - w, -w / 2, w, w),
                             Qt.red)
        painter.rotate(-90)

    def _draw_arrow(color, rotation, height):
        painter.save()
        painter.rotate(-90 * rotation)
        s = (DEBUG_IMAGE_SIZE * (1 - 2 * w)) / DEBUG_IMAGE_SIZE
        painter.scale(s, height * s)
        painter.translate(-.5, -.5)
        painter.fillPath(ARROW_PATH, color)
        painter.restore()

    signs = np.nonzero(array[4:])[0]
    if len(signs) > 0:
        i = signs[0]
        _draw_arrow(QColor.fromHsvF(0, 0, array[4 + i]), i, 1)

    def a_to_dir(a):
        return Maze._offsets_inv[a]

    correct_direction = a_to_dir(correct_action)
    if correct_action != selected_action:
        _draw_arrow(Qt.blue, correct_direction.value, .5)

    if selected_action is not None:
        selected_direction = a_to_dir(selected_action)
        if selected_direction == correct_direction:
            _draw_arrow(Qt.green, correct_direction.value, .5)
        else:
            _draw_arrow(Qt.red, selected_direction.value, .5)

    painter.end()
    return image


def __debug_draw_inputs(folder: Path, inputs, outputs: Optional = None):
    i_type_list = {t: i for i, t in enumerate([None, SignType.LURE,
                                               SignType.CLUE, SignType.TRAP])}
    inputs = sorted(inputs, key=lambda t: i_type_list[t[2]])
    pprint.pprint(inputs)

    n_inputs = len(inputs)
    i_digits = math.ceil(math.log10(n_inputs))
    cols = math.ceil(math.sqrt(n_inputs) * 16/9)
    rows = math.ceil(n_inputs / cols)

    margin = 4
    big_image = QImage(cols * (DEBUG_IMAGE_SIZE + margin),
                       rows * (DEBUG_IMAGE_SIZE + margin),
                       QImage.Format_ARGB32)
    big_image.fill(Qt.transparent)
    painter = QPainter(big_image)

    for i, ((arr, act, _), o) in \
            enumerate(itertools.zip_longest(inputs, outputs or [])):
        img = __debug_draw_input(arr, act, o)
        img.save(str(folder.joinpath(f"{i:0{i_digits}d}.png")))

        x, y = i % cols, i // cols
        painter.drawImage(QPointF(x * (DEBUG_IMAGE_SIZE + margin) + .5 * margin,
                                  y * (DEBUG_IMAGE_SIZE + margin) + .5 * margin),
                          img)

    painter.end()
    big_image.save(str(folder.joinpath("inputs.png")))


def generate_inputs():
    inputs: List[Tuple[np.ndarray, Action, Optional[SignType]]] = []

    def array():
        return np.zeros(8)

    def i_to_dir(i_):
        return Maze._offsets[Maze.Direction(i_)]

    # First the dead ends
    for i in range(4):
        a = array()
        a[:4] = 1
        a[i] = 0
        inputs.append((a, i_to_dir(i), None))
        a = a.copy()
        a[i] = .5
        inputs.append((a, i_to_dir(i), None))

    # Then the corridors
    for i in range(3):
        for j in range(i + 1, 4):
            dirs = set(range(4)) - {i, j}
            for d in dirs:
                a = array()
                a[i] = 1
                a[j] = 1
                a[d] = .5
                i_sol = next(iter(dirs - {d}))
                sol = i_to_dir(i_sol)
                inputs.append((a, sol, None))
                for lure in iter(set(range(4)) - {i_sol}):
                    a_ = array()
                    a_[:] = a[:]
                    a_[4:] = 0
                    a_[lure + 4] = .25
                    inputs.append((a_, sol, SignType.LURE))

    # And the intersections
    for i in range(4):
        for j in iter(set(range(4)) - {i}):
            for k in iter(set(range(4)) - {i}):
                a = array()
                a[i] = 1
                a[k] = .5
                a[j + 4] = 1
                inputs.append((a, i_to_dir(j), SignType.CLUE))

                if j != k:
                    a = a.copy()
                    a[j + 4] = .5
                    i_sol = next(iter(set(range(4)) - {i, j, k}))
                    inputs.append((a, i_to_dir(i_sol), SignType.TRAP))

    if not (folder := Path("results/re-eval_data/inputs")).exists():
        folder.mkdir(parents=True)
        __debug_draw_inputs(folder, inputs)

    return inputs


def evaluate_inputs(inputs, controller, folder):
    counts = {k: [0, 0] for k in [None, *[s for s in SignType]]}
    outputs = []
    for i, a, t in inputs:
        a_: Vec = controller(i)
        # print(i, a, t)
        # print(f"> {a} =?= {a_}: {a == a_}")
        counts[t][0] += int(a_ == a)
        counts[t][1] += 1
        outputs.append(tuple(a_))

    folder.mkdir(parents=True, exist_ok=True)
    __debug_draw_inputs(folder, inputs, outputs)

    data = {"Empty" if k is None else k.value:
            v[0] / v[1] for k, v in counts.items()}
    data["All"] = np.average(list(data.values()))
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["Success"])
    df.index.name = "Signs"
    return df


def swarmed_violinplot(common_args: dict,
                       violin_args: dict = VIOLIN_ARGS,
                       swarm_args: dict = SWARM_ARGS,
                       analyze_args: Optional[dict] = None,
                       analyze_value: Optional[float | str] = None):

    swarm_args = swarm_args.copy()
    swarm_args.pop('kind')

    fig, ax = plt.subplots()
    seaborn.swarmplot(**common_args, ax=ax, **swarm_args)
    handles = ax.legend_.legendHandles
    labels = [text.get_text() for text in ax.legend_.texts]

    seaborn.violinplot(**common_args, **violin_args, ax=ax)

    if analyze_args and analyze_value:
        analyze({analyze_value: ax}, **analyze_args, **common_args)

    move_legend(fig, ax=ax, hdl=handles, lbl=labels, loc='best')

    return fig, ax


def analyze(axes_dict, order, hue_order, data,
            pivot_columns, data_column,
            **plot_args):
    pairs = [
        *[((k, "a2c"), (k, "ppo")) for k in order],
        # *[((X_ORDER[i], k3), (X_ORDER[j], k3)) for k3 in ["a2c", "ppo"]
        #   for i in range(2) for j in range(i+1, 3)]
        *[((order[i], k3), (order[j], k3)) for k3 in hue_order
          for i in range(2) for j in range(i+1, 3)]
    ]

    for name, ax in axes_dict.items():
        # ax.set_ylim(None, 1)

        def sample(t, a):
            data_mask = (data[pivot_columns] == name)
            if isinstance(data_mask, pd.DataFrame):
                data_mask = data_mask.all(axis=1)
            df__ = data[data_mask
                        & (data.Trainer == t)
                        & (data.Algo == a)][data_column]
            # print(f"sample({pivot_columns}={name}, trainer={t}, algo={a}):\n{df__}")
            return df__

        mw_stats = [
            mannwhitneyu(sample(t1, a1), sample(t2, a2)).pvalue
            for ((t1, a1), (t2, a2)) in pairs
        ]
        mw_c_stats = false_discovery_control(mw_stats)
        tt_stats = [
            ttest_ind(sample(t1, a1), sample(t2, a2)).pvalue
            for ((t1, a1), (t2, a2)) in pairs
        ]
        tt_c_stats = false_discovery_control(tt_stats)
        stats = tt_c_stats
        positive = [(s <= .04) for s in stats]
        filtered_pairs = [pair for pair, p in zip(pairs, positive) if p]

        print("#" * 80)
        print(f"## {pivot_columns} = {name} ({data_column})")
        print("#" * 10)
        print("   ", " ".join([f"{s:^7s}" for s in ["mw", "mw_c", "tt", "tt_c"]]))
        for i, ((kl, kr), v, p) in enumerate(zip(pairs, stats, positive)):
            msg = (f"[{'X' if p else ' '}] "
                   + " ".join([f"{lst[i]:7.2g}" for lst in [mw_stats, mw_c_stats, tt_stats, tt_c_stats]])
                   + " "
                   + " vs ".join(["/".join(k) for k in [kl, kr]]))
            if p:
                msg = f"\033[1m{msg}\033[0m"
            print(msg)

        if len(filtered_pairs) > 0:
            annot = Annotator(ax=ax, pairs=filtered_pairs,
                              order=order, hue_order=hue_order, data=data,
                              **plot_args)
            annot.configure(text_format='star', verbose=0,
                            )
            annot.set_pvalues([s for s, p in zip(stats, positive) if p])
            annot.annotate(line_offset=.1, line_offset_to_group=None)
        print("=" * 80)


@dataclass
class Options:
    models: List[Path] = field(default_factory=list)

    data_folder: Path = Path("results/re-eval_data/")
    out_folder: Path = Path("results/re-eval/")

    size: int = 256

    force_generate: bool = False
    force_eval: bool = False
    summarize: bool = True
    plot_inputs: bool = True
    plot_evals: bool = True
    plot_scatters: bool = True
    plot_results: bool = True

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("models", metavar='model', nargs='*',
                            action='extend', type=Path,
                            help="Model to evaluate (can be repeated)")

        parser.add_argument("--data", dest="data_folder",
                            metavar="DIR", type=Path,
                            help="Where to look for re-eval data (mazes, ...)")

        parser.add_argument("--out", dest="out_folder",
                            metavar="DIR", type=Path,
                            help="Where to store the resulting data")

        parser.add_argument("--force-generate",
                            action='store_true',
                            help="Force sampling/generation of test mazes")

        parser.add_argument("--force-eval",
                            action='store_true',
                            help="Force evaluation of provided models")

        parser.add_argument("--no-summarize", dest='summarize',
                            action='store_false',
                            help="Prevent auto-merging of model stats into a"
                                 " wide-format dataframes")

        parser.add_argument("--no-plot-inputs", dest='plot_inputs',
                            action='store_false',
                            help="Do not plot inputs-related graph")

        parser.add_argument("--no-plot-evals", dest='plot_evals',
                            action='store_false',
                            help="Do not plot re-eval performance graphs")

        parser.add_argument("--no-plot-scatters",
                            dest='plot_scatters',
                            action='store_false',
                            help="Do not plot cross-performance graphs")


def main():
    args = Options()
    parser = argparse.ArgumentParser(
        description="Posterior evaluator for sb3-like trainings")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    pprint.pprint(args)
    args.models = sorted(list(set(m_ for m in args.models for m_ in
                                  ([m] if "*" not in str(m) else
                                   [Path(p) for p in glob.glob(str(m))]))))
    print("Got", len(args.models), "agents to test")

    mazes_file = args.data_folder.joinpath("mazes.csv")
    if args.force_generate or not mazes_file.exists():
        _generate_mazes(mazes_file)
    mazes = pd.read_csv(mazes_file, index_col=list(range(4)))

    simulation = Simulation(
        Maze.from_string("5x5"),
        Robot.BuildData(inputs=InputType.DISCRETE,
                        outputs=OutputType.DISCRETE),
        save_trajectory=True
    )

    n_models, n_mazes = len(args.models), len(mazes.index) * 4
    pb = tqdm(total=n_models * n_mazes, desc="Evaluating")

    with CV2QTGuard():
        app = QApplication([])

    start = time.perf_counter()

    inputs = generate_inputs()

    stats_files = []
    for controller_file in args.models:
        if not controller_file.exists():
            print(f"'{controller_file}' does not exist. Skipping")
            pb.update(n_mazes)
            continue

        out_folder = args.out_folder.joinpath(controller_file.parent)
        stats_file = out_folder.joinpath("stats.csv")
        stats_files.append(stats_file)

        controller = load(controller_file)
        if not args.force_eval and out_folder.exists():
            pb.update(n_mazes)
            continue

        out_folder.mkdir(parents=True, exist_ok=True)

        inputs_df = evaluate_inputs(inputs, controller,
                                    out_folder.joinpath("inputs"))
        inputs_df.to_csv(out_folder.joinpath("inputs.csv"))

        infos_df = None
        for index in mazes.index:
            t, it, m, jm = index
            stats, images = [], []
            for bd in Maze.bd_from_string(m).all_permutations():
                simulation.reset(maze=Maze.generate(bd))
                while not simulation.done():
                    simulation.step(controller(simulation.observations))
                stats.append(simulation.infos())

                img = MazeWidget.plot_trajectory(
                    simulation=simulation,
                    size=256, trajectory=simulation.trajectory,
                    config=dict(
                        solution=True,
                        robot=False,
                        dark=True
                    ),
                    path=None
                )
                images.append(img)
                pb.update(1)

            avg_stats = recurse_avg_dict(stats)
            if infos_df is None:
                infos_df = pd.DataFrame(columns=list(avg_stats.keys()),
                                        index=mazes.index)
            infos_df.loc[index, :] = list(avg_stats.values())

            t_file = out_folder.joinpath(f"{it}__{jm}__{m}.png")
            big_image = merge_images(images)
            big_image.save(str(t_file))

        infos_df.to_csv(stats_file)

        files = list(out_folder.glob("*__*__*.png"))
        files = sorted(files,
                       key=lambda p:
                       tuple(int(i) for i in reversed(p.name.split('__')[:2])))
        subprocess.Popen(["montage", "-geometry", "+10+10", "-tile", "6x",
                          *files,
                          f"{out_folder}/trajectories.png"],
                         stdout=sys.__stdout__, stderr=sys.__stderr__)

    pb.close()
    print(f"Reevaluated {len(args.models)} agents in {_pretty_delta(start)}")

    big_stats_csv = args.out_folder.joinpath("stats.csv")
    big_inputs_csv = args.out_folder.joinpath("inputs.csv")
    summary_csv = args.out_folder.joinpath("summary.csv")
    if args.summarize:
        start = time.perf_counter()
        big_stats_df, big_inputs_df = pd.DataFrame(), pd.DataFrame()
        key_regex = re.compile(r'.*/([^/]*)/([^/]*run[0-9]*)/.*')
        for f in stats_files:
            key = key_regex.match(str(f))
            trainer, algo = key[1].split('-')

            def add_columns(df_, i=0):
                df_["Trainer"] = trainer
                df_["Algo"] = algo
                df_["Replicate"] = key[2]
                c = df_.columns.to_list()
                c = c[0:i] + c[-3:] + c[i:-3]
                return df_[c]

            df = add_columns(pd.read_csv(f), 4)
            big_stats_df = pd.concat([big_stats_df, df], ignore_index=True)

            df = add_columns(pd.read_csv(f.with_name("inputs.csv")))
            big_inputs_df = pd.concat([big_inputs_df, df], ignore_index=True)

        def agg(col, mapping):
            return v(col) \
                if (v := mapping.get(col.name, None)) is not None \
                else col.mean()

        rows_marginals = (
            big_stats_df.groupby(['Trainer', 'Algo', 'Replicate', 'MI'],
                                 as_index=False)
            .agg(partial(agg, mapping=dict(
                Name=lambda _: np.nan,
                Type=lambda c: "r_marginal",
                TI=lambda c: c.max() + 1,
                MI=lambda c: c.mean()))))
        cols_marginals = (
            big_stats_df.groupby(['Trainer', 'Algo', 'Replicate', 'TI'],
                                 as_index=False)
            .agg(partial(agg, mapping=dict(
                Name=lambda _: np.nan,
                Type=lambda c: "c_marginal",
                TI=lambda c: c.mean(),
                MI=lambda c: c.max() + 1))))

        marginals = (
            big_stats_df.groupby(['Trainer', 'Algo', 'Replicate'],
                                 as_index=False)
            .agg(partial(agg, mapping=dict(
                Name=lambda _: np.nan,
                Type=lambda _: "all",
                TI=lambda c: c.max() + 1,
                MI=lambda c: c.max() + 1))))

        big_stats_df = pd.concat([big_stats_df,
                                  rows_marginals, cols_marginals,
                                  marginals])

        big_stats_df.insert(9, 'speed',
                            big_stats_df['len'] / big_stats_df['steps'])
        big_stats_df.sort_values(by=list(big_stats_df.columns[:4]), axis=0,
                                 inplace=True, ignore_index=True)
        big_stats_df.to_csv(big_stats_csv, index=False)

        big_inputs_df.to_csv(big_inputs_csv, index=False)

        inputs_summary = (
            big_inputs_df.loc[big_inputs_df["Signs"] == "All"]
            .drop("Signs", axis=1).rename(columns={'Replicate': 'Run'}))
        inputs_summary.set_index(inputs_summary.columns[:3].tolist(),
                                 inplace=True)
        inputs_summary["Rank"] = (
            inputs_summary.rank(method='dense', ascending=False).astype(int))
        inputs_summary.sort_values(by=["Rank", *inputs_summary.index.names],
                                   inplace=True)
        inputs_summary.set_index(pd.Series(range(len(inputs_summary)),
                                           name="I"),
                                 append=True, inplace=True)
        inputs_summary.to_csv(args.out_folder.joinpath("inputs_ranks.csv"))

        summary_index = ["Trainer", "Algo", "Replicate"]
        summary_df = [
            big_stats_df[big_stats_df.Type == 'all']
            .drop(['Type', 'TI', 'Name', 'MI'],
                  axis='columns').set_index(summary_index),
            big_inputs_df.pivot(index=summary_index,
                                columns="Signs", values="Success")
        ]
        summary_df = summary_df[0].join(summary_df[1], on=summary_index)
        summary_df.drop(["time", "failure", "done", "len", "steps",
                         *[f"errors/{s}" for s in ["clue", "lure", "trap"]]],
                        axis=1, inplace=True)
        rank_df = (summary_df.rank(ascending=False).astype(int)
                   .rename(columns={k: f"{k}_r" for k in summary_df.columns}))
        summary_df = summary_df.join(rank_df)
        summary_df = summary_df[sorted(summary_df.columns)]
        summary_df.to_csv(summary_csv)

        print(f"Aggregated data in {_pretty_delta(start)}")

    else:
        big_stats_df = pd.read_csv(big_stats_csv)
        big_inputs_df = pd.read_csv(big_inputs_csv)
        summary_df = pd.read_csv(summary_csv)

    if any([args.plot_inputs, args.plot_evals, args.plot_results]):
        start = time.perf_counter()

        set_seaborn_style()

        inputs_facets_args = dict(
            data=big_inputs_df[big_inputs_df["Signs"] != "All"],
            x="Trainer", col='Signs', y="Success", hue="Algo",
            order=X_ORDER, hue_order=HUE_ORDER)

        inputs_average_args = inputs_facets_args.copy()
        inputs_average_args['data'] = (
            big_inputs_df[big_inputs_df["Signs"] == "All"])
        inputs_average_args.pop('col')

        inputs_analyze_args = dict(pivot_columns="Signs",
                                   data_column="Success")

        def plot_inputs_per_signs(_analyze, n_cols=None):
            _g = seaborn.catplot(**SWARM_ARGS, col_wrap=n_cols,
                                 **inputs_facets_args)
            _g.map_dataframe(seaborn.violinplot,
                             **VIOLIN_ARGS, **inputs_facets_args)
            if _analyze:
                analyze(axes_dict=_g.axes_dict, **inputs_analyze_args,
                        **inputs_facets_args)
            move_legend(_g)
            plt.subplots_adjust(hspace=0.2, wspace=0.05)
            return _g.figure

        def plot_inputs_average():
            return swarmed_violinplot(common_args=inputs_average_args,
                                      analyze_args=inputs_analyze_args,
                                      analyze_value="All")

        if args.plot_inputs:
            inputs_summary_plot = args.out_folder.joinpath("inputs.pdf")
            with PdfPages(inputs_summary_plot) as pdf:
                fig = plot_inputs_per_signs(_analyze=True)
                pdf.savefig(fig)
                fig, ax = plot_inputs_average()
                pdf.savefig(fig)

            print("Saved inputs summary plot to", inputs_summary_plot)

        if args.plot_evals:
            stats_summary_plot = args.out_folder.joinpath("stats.pdf")
            big_stats_df.drop("Name", axis='columns', inplace=True)

            columns = [c for c in big_stats_df.columns if c[0].lower() == c[0]]
            for column in ["time", "failure", "done", "len", "steps"]:
                columns.remove(column)

            print("Plotting data for columns:", " ".join(columns))
            with PdfPages(stats_summary_plot) as pdf:
                c_names = dict(big_stats_df[["TI", "Type"]]
                               .sort_values(by="TI").values)
                r_names = {0: "Min", 1: "Median", 2: "Max", 3: "c_marginal"}
                for column in columns:
                    common_args = dict(
                        data=big_stats_df,
                        x="Trainer", y=column, hue="Algo",
                        order=X_ORDER, hue_order=HUE_ORDER
                    )
                    plot = seaborn.catplot(**common_args,
                                           **SWARM_ARGS,
                                           col="TI", row="MI",
                                           margin_titles=True)

                    plot.map_dataframe(seaborn.violinplot,
                                       **VIOLIN_ARGS, **common_args)

                    # analyze(g.axes_dict, **common_args)

                    plot.figure.suptitle(column, fontsize='x-large')
                    rows, cols = plot.axes.shape
                    for i, j in np.ndindex(plot.axes.shape):
                        ax: Axes = plot.axes[i, j]
                        if title := ax.get_title():
                            key = int(title.split(' = ')[-1])
                            ax.set_title(c_names.get(key, "???"))
                        if text := next((c for c in ax.get_children()
                                         if isinstance(c, Annotation)), None):
                            key = int(text.get_text().split(' = ')[-1])
                            text.set_text(r_names.get(key, "!!!"))

                        last_row, last_col = (i == rows - 1), (j == cols - 1)
                        if last_row or last_col:
                            r, g, b, a = ax.get_facecolor()
                            r, g, b = [v * .9 ** (int(last_row) + int(last_col))
                                       for v in (r, g, b)]
                            ax.set_facecolor((r, g, b, a))

                        if last_row and last_col:
                            analyze({(rows - 1, cols - 1): ax},
                                    pivot_columns=["MI", "TI"],
                                    data_column=column,
                                    **common_args)
                    move_legend(plot)

                    pdf.savefig(plot.figure)
            print("Saved stats summary plot to", stats_summary_plot)

        if False and args.plot_scatters:
            results_plot = args.out_folder.joinpath("scatters.pdf")
            with PdfPages(results_plot) as pdf:
                print("Plotting scatters")
                print(summary_df.to_string(max_rows=100, max_cols=100))

                for lhs in ["All", "Clue", "Trap"]:
                    for rhs in ["pretty_reward", "success"]:
                        g = seaborn.displot(
                            kind='kde',
                            data=summary_df, x=lhs, y=rhs,
                            hue="Trainer", col="Algo",
                            hue_order=X_ORDER,
                            fill=True, alpha=.3, legend=False)
                        g.map_dataframe(
                            seaborn.scatterplot,
                            data=summary_df, x=lhs, y=rhs,
                            hue="Trainer", #col="Algo",
                            style='Replicate',
                            hue_order=X_ORDER, legend=True)
                        g.add_legend()

                        g.figure.suptitle(f"{lhs} / {rhs}")

                        pdf.savefig(g.figure)

            # complexity_plot = args.out_folder.joinpath("complexity.pdf")
            # with (PdfPages(complexity_plot) as pdf):
            #     print("Plotting complexity")
            #     mazes_df = pd.read_csv(
            #         args.out_folder.parent
            #         .joinpath("train_summary/edhucat_mazes.csv"),
            #         index_col=['Algo', 'Run', 'I', 'T']
            #     )
            #
            #     df = big_inputs_df[(big_inputs_df.Signs == 'All') &
            #                        (big_inputs_df.Trainer == 'edhucat')]
            #     df.set_index(keys=["Algo", "Replicate"], inplace=True)
            #     for (a, r), v in df['Success'].items():
            #         mazes_df.loc[(a, int(r[-1]), slice(None), slice(None)), "Signs"] = v
            #
            #     df = big_stats_df[(big_stats_df.Type == 'all') &
            #                       (big_stats_df.Trainer == 'edhucat')]
            #     df.set_index(keys=["Algo", "Replicate"], inplace=True)
            #     for (a, r), v in df[["pretty_reward", "success"]].iterrows():
            #         mazes_df.loc[(a, int(r[-1]), slice(None), slice(None)),
            #                      ["Reward", "Success"]] = v.values
            #
            #     mazes_df.rename_axis(index={'I': "Stage"}, inplace=True)
            #
            #     cmap = "viridis"
            #
            #     interpolation_df = pd.read_csv(args.data_folder.joinpath(
            #         "../interpolation_data/traps-median/mazes_stats.csv"))
            #     interpolation_df.replace({'Set': {0: 'Eval', 1: 'Train'}},
            #                              inplace=True)
            #     interpolation_df.rename(columns={"ID": "Stage", "Set": "T"},
            #                             inplace=True)
            #
            #     for m_col in ["Emin", "Emax", "path", "intersections",
            #                   "clues", "lures", "traps"]:
            #         v_min, v_max = np.quantile(mazes_df[m_col], [0, 1])
            #         mazes_df[m_col] += np.random.normal(
            #             0, (v_max - v_min) / 20, len(mazes_df))
            #
            #         for h_col in ['Signs', "Success", "Reward"]:
            #
            #             norm = plt.Normalize()
            #             sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            #
            #             g = seaborn.relplot(kind='line',
            #                                 data=mazes_df,
            #                                 x="Stage", y=m_col,
            #                                 col='Algo', row="T",
            #                                 style='Run', dashes=False,
            #                                 hue=h_col, hue_norm=norm,
            #                                 palette=cmap,
            #                                 legend=False)
            #             for ax in g.axes.flatten():
            #                 t = ax.get_title().split(' | ')[0]\
            #                       .split('=')[1].strip()
            #                 seaborn.lineplot(
            #                     ax=ax,
            #                     data=interpolation_df[interpolation_df['T'] == t],
            #                     x="Stage", y=m_col,
            #                     color='gray', dashes=(1, 1))
            #
            #             g.fig.subplots_adjust(top=.95, bottom=.05)
            #             g.fig.colorbar(sm, ax=g.axes, cmap=cmap,
            #                            fraction=.05, pad=.025,
            #                            label=h_col)
            #             g.fig.legend(handles=[
            #                 Line2D([0], [0], color='black',
            #                        label='EDHuCAT'),
            #                 Line2D([0], [0], color='gray',
            #                        dashes=(1, 1), label='Interpolation')])
            #
            #             g.figure.suptitle(f"{m_col} / {h_col}")
            #             pdf.savefig(g.figure)
            #             plt.close(g.figure)

        if args.plot_results:
            results_plot = args.out_folder.parent.joinpath("summary.pdf")
            with PdfPages(results_plot) as pdf:

                def prettify(_ax: Axes, _y_label: Optional[str]):
                    if _y_label:
                        _ax.set_ylabel(_y_label)
                    _ax.set_xticklabels(["Direct", "Interpolation", "EDHuCAT"])

                # ==============================================================
                # Plot the maze navigation stats

                re_eval_df = big_stats_df[(big_stats_df.MI == 3)
                                          & (big_stats_df.TI == 6)]
                re_eval_df.loc[:, 'success'] *= 100
                common_args = dict(data=re_eval_df, x="Trainer", hue="Algo",
                                   order=X_ORDER, hue_order=HUE_ORDER)
                for column, name in [("pretty_reward", "Normalized reward"),
                                     ("success", "Success (%)")]:
                    common_args['y'] = column
                    fig, ax = swarmed_violinplot(
                        common_args,
                        analyze_args=dict(pivot_columns="Type",
                                          data_column=column),
                        analyze_value="all")
                    prettify(_ax=ax, _y_label=name)
                    pdf.savefig(fig)

                # ==============================================================
                # Plot the input response

                y_label = "Success (%)"

                inputs_facets_args['data'].loc[:, 'Success'] *= 100
                fig: Figure = plot_inputs_per_signs(_analyze=False, n_cols=2)
                for ax in fig.axes:
                    prettify(_ax=ax, _y_label=y_label)
                fig.set_size_inches(FIG_SIZE_INCHES)
                pdf.savefig(fig)

                inputs_average_args['data'].loc[:, 'Success'] *= 100
                fig, ax = plot_inputs_average()
                prettify(_ax=ax, _y_label=y_label)
                pdf.savefig(fig)

                # ==============================================================
                # Plot the edhucat stuff

                plt.rcParams["figure.figsize"] = (
                    FIG_SIZE_INCHES[0] / 3, FIG_SIZE_INCHES[1])

                def df_filter(df_, c):
                    if "Trainer" in df_.columns:
                        df_ = df_[df_.Trainer == 'edhucat']
                    return df_.set_index(keys=["Algo", "Replicate"])[c]

                edhucat_summary = df_filter(re_eval_df,
                                            ["success", "pretty_reward"])
                edhucat_summary = edhucat_summary.join(
                    df_filter(inputs_average_args['data'], "Success")
                ).join(
                    df_filter(
                        pd.read_csv(args.out_folder.parent
                                    .joinpath("train_summary")
                                    .joinpath("edhucat_strategies.csv")),
                        ["Strategy"])
                ).reset_index()
                print(edhucat_summary.to_string(max_cols=100, max_rows=100))

                for y, name in [("pretty_reward", "Average normalized reward"),
                                ("success", "Goal reaching rate (%)"),
                                ("Success", "Inputs processing rate (%)")]:
                    fig, ax = plt.subplots()
                    seaborn.swarmplot(
                        data=edhucat_summary, ax=ax,
                        x="Strategy", y=y, hue="Algo",
                        order=sorted(edhucat_summary['Strategy'].values)
                    )
                    ax.set_ylabel(name)
                    seaborn.move_legend(obj=ax, loc="best", title=None)

                    pdf.savefig(ax.figure)

            print("Saved results plot to", results_plot)

        print(f"Plotted summaries in {_pretty_delta(start)}")


if __name__ == '__main__':
    main()
