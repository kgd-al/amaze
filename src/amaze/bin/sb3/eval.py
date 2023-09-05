#!/usr/bin/env python3
import argparse
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
from pathlib import Path
from typing import List, Tuple, Optional

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF
from PyQt5.QtGui import QImage, QPainter, QColor
from PyQt5.QtWidgets import QApplication
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from tqdm import tqdm

from amaze.bin import maze_viewer
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
            painter.fillRect(QRectF(.5 - w, -w/2, w, w),
                             Qt.red)
        painter.rotate(-90)

    def _draw_arrow(color, rotation, height):
        painter.save()
        painter.rotate(-90*rotation)
        s = (DEBUG_IMAGE_SIZE * (1 - 2*w)) / DEBUG_IMAGE_SIZE
        painter.scale(s, height * s)
        painter.translate(-.5, -.5)
        painter.fillPath(ARROW_PATH, color)
        painter.restore()

    signs = np.nonzero(array[4:])[0]
    if len(signs) > 0:
        i = signs[0]
        _draw_arrow(QColor.fromHsvF(0, 0, array[4+i]), i, 1)

    def a_to_dir(a): return Maze._offsets_inv[a]
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
    n_inputs = len(inputs)
    i_digits = math.ceil(math.log10(n_inputs))
    cols = math.ceil(math.sqrt(n_inputs))
    rows = math.ceil(n_inputs / cols)

    margin = 4
    big_image = QImage(cols * (DEBUG_IMAGE_SIZE + margin),
                       rows * (DEBUG_IMAGE_SIZE + margin),
                       QImage.Format_RGB32)
    big_image.fill(Qt.darkGray)
    painter = QPainter(big_image)

    for i, ((arr, act, _), o) in \
            enumerate(itertools.zip_longest(inputs, outputs or [])):
        img = __debug_draw_input(arr, act, o)
        img.save(str(folder.joinpath(f"{i:0{i_digits}d}.png")))

        x, y = i % cols, i // cols
        painter.drawImage(QPointF(x * (DEBUG_IMAGE_SIZE+margin) + .5*margin,
                                  y * (DEBUG_IMAGE_SIZE+margin) + .5*margin),
                          img)

    painter.end()
    big_image.save(str(folder.joinpath("inputs.png")))


def generate_inputs():
    inputs: List[Tuple[np.ndarray, Action, Optional[SignType]]] = []
    def array(): return np.zeros(8)
    def i_to_dir(i_): return Maze._offsets[Maze.Direction(i_)]

    # First the dead ends
    for i in range(4):
        a = array()
        a[:4] = 1
        inputs.append((a, i_to_dir(i), None))
        a = a.copy()
        a[i] = .5
        inputs.append((a, i_to_dir(i), None))

    # Then the corridors
    for i in range(3):
        for j in range(i+1, 4):
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
                    a_[lure+4] = .25
                    inputs.append((a_, sol, SignType.LURE))

    # And the intersections
    for i in range(4):
        for j in iter(set(range(4)) - {i}):
            for k in iter(set(range(4)) - {i}):
                a = array()
                a[i] = 1
                a[k] = .5
                a[j+4] = 1
                inputs.append((a, i_to_dir(j), SignType.CLUE))

                if j != k:
                    a = a.copy()
                    a[j+4] = .5
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
            v[0]/v[1] for k, v in counts.items()}
    data["All"] = np.average(list(data.values()))
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["Success"])
    df.index.name = "Signs"
    return df


def analyze(axes_dict, order, hue_order, data, **plot_args):
    pairs = [
        *[((k, "a2c"), (k, "ppo")) for k in order],
        # *[((x_order[i], k3), (x_order[j], k3)) for k3 in ["a2c", "ppo"]
        #   for i in range(2) for j in range(i+1, 3)]
        *[((order[i], k3), ("edhucat", k3)) for k3 in hue_order
          for i in range(2)]
    ]

    for name, ax in axes_dict.items():
        # ax.set_ylim(None, 1)

        def sample(t, a):
            return data[(data.Signs == name)
                        & (data.Trainer == t)
                        & (data.Algo == a)]["Success"]

        stats = [
            mannwhitneyu(sample(t1, a1), sample(t2, a2)).pvalue
            for ((t1, a1), (t2, a2)) in pairs
        ]
        positive = [s <= .05 for s in stats]
        filtered_pairs = [pair for pair, p in zip(pairs, positive) if p]

        print("="*80)
        print("==", name)
        print("="*10)
        for (kl, kr), v, p in zip(pairs, stats, positive):
            print("[X]" if p else "[-]",
                  " vs ".join(["/".join(k) for k in [kl, kr]]))

        if len(filtered_pairs) > 0:
            annot = Annotator(ax=ax, pairs=filtered_pairs,
                              order=order, hue_order=hue_order, data=data,
                              **plot_args)
            annot.configure(text_format='star', verbose=0,
                            )
            annot.set_pvalues([s for s, p in zip(stats, positive) if p])
            annot.annotate(line_offset=.1, line_offset_to_group=None)
        print("="*80)


def move_legend(plot, ax=None, hdl=None, lbl=None):
    l_args = dict(ncol=2, frameon=True, title=None,
                  handles=hdl, labels=lbl)
    if isinstance(plot, Figure):
        ax.legend(**l_args)
    else:
        seaborn.move_legend(obj=plot,
                            loc="lower center", bbox_to_anchor=(.5, 0),
                            **l_args)
    # plt.subplots_adjust(top=.6)
    plt.tight_layout()


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
    plot_eval: bool = True

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

        parser.add_argument("--no-plot-eval", dest='plot_eval',
                            action='store_false',
                            help="Do not plot re-eval performance graphs")


def main():
    args = Options()
    parser = argparse.ArgumentParser(
        description="Posterior evaluator for sb3-like trainings")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    pprint.pprint(args)

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

        print(f"Aggregated data in {_pretty_delta(start)}")

    else:
        big_stats_df = pd.read_csv(big_stats_csv)
        big_inputs_df = pd.read_csv(big_inputs_csv)

    if args.plot_inputs or args.plot_eval:
        start = time.perf_counter()
        c_names = dict(big_stats_df[["TI", "Type"]].sort_values(by="TI").values)
        stats_summary_plot = args.out_folder.joinpath("stats.pdf")

        swarm_args = dict(kind='swarm', palette='deep', dodge=True)
        violin_args = dict(kind='violin', inner=None, cut=0,
                           scale='width', palette='pastel')
        x_order = ["direct", "interpolation", "edhucat"]
        hue_order = ["a2c", "ppo"]
        seaborn.set_style("darkgrid")

        if args.plot_inputs:
            inputs_summary_plot = args.out_folder.joinpath("inputs.pdf")
            with PdfPages(inputs_summary_plot) as pdf:

                common_args = dict(
                    data=big_inputs_df[big_inputs_df["Signs"] != "All"],
                    x="Trainer", col='Signs', y="Success", hue="Algo",
                    order=x_order, hue_order=hue_order)
                g = seaborn.catplot(**swarm_args, **common_args)
                g.map_dataframe(seaborn.violinplot,
                                **violin_args, **common_args)
                analyze(g.axes_dict, **common_args)
                move_legend(g)
                pdf.savefig(g.figure)

                common_args.update(dict(
                    data=big_inputs_df[big_inputs_df["Signs"] == "All"],
                ))
                common_args.pop('col')
                fig, ax = plt.subplots()
                seaborn.swarmplot(**common_args, ax=ax,
                                  **{k: v for k, v in swarm_args.items()
                                     if k != "kind"})
                handles = ax.legend_.legendHandles
                labels = [text.get_text() for text in ax.legend_.texts]

                seaborn.violinplot(**common_args, **violin_args, ax=ax)

                analyze({"All": ax}, **common_args)
                move_legend(fig, ax=ax, hdl=handles, lbl=labels)
                fig.tight_layout()
                pdf.savefig(fig)

            print("Saved inputs summary plot to", inputs_summary_plot)

        if args.plot_eval:
            columns = [c for c in big_stats_df.columns if c[0].lower() == c[0]]
            for c in ["time", "failure", "done", "len", "steps"]:
                columns.remove(c)

            def foo(x):
                print(f"x='{x}'")
                return x

            print(big_stats_df)
            big_stats_df['speed'] = big_stats_df['len'] / big_stats_df['steps']
            print(big_stats_df.groupby(['Trainer', 'Algo', 'Replicate']).agg(foo))

            print("Plotting data for columns:", " ".join(columns))

            with PdfPages(stats_summary_plot) as pdf:
                for c in columns:
                    common_args = dict(
                        data=big_stats_df,
                        x="Trainer", y=c, hue="Algo",
                        order=x_order, hue_order=hue_order
                    )
                    g = seaborn.catplot(**common_args,
                                        **swarm_args, s=4,
                                        col="TI", row="MI", margin_titles=True)

                    g.map_dataframe(seaborn.violinplot,
                                    **violin_args, **common_args)

                    # analyze(g.axes_dict, **common_args)

                    g.figure.suptitle(c, fontsize='x-large')
                    for ax, name in zip(g.axes[0], c_names.values()):
                        ax.set_title(name)
                    move_legend(g)

                    pdf.savefig(g.figure)
            print("Saved stats summary plot to", stats_summary_plot)

        print(f"Plotted summary in {_pretty_delta(start)}")


if __name__ == '__main__':
    main()
