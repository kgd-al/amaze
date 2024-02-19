#!/usr/bin/env python3

import argparse
import copy
import functools
import logging
import math
import pprint
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union, List, Tuple

import humanize
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QLineF, QSignalBlocker
from PyQt5.QtGui import QIcon, QImage, QPainter, QFont
from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QDialog, QApplication,
                             QButtonGroup, QSizePolicy, QPushButton, QWidget, QGridLayout, QLabel, QLayout)
from stable_baselines3 import SAC, A2C, DQN, PPO, TD3
from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from amaze.sb3.callbacks import TensorboardCallback
from amaze.sb3.maze_env import MazeEnv
from amaze.sb3.utils import CV2QTGuard
from amaze.simu.env.maze import Maze
from amaze.simu.robot import Robot, InputType, OutputType
from amaze.simu.simulation import Simulation
from amaze.utils.tee import Tee
from amaze.visu.resources import SignType
from amaze.visu.viewer import MainWindow

logger = logging.getLogger("edhucat-main")


class OverwriteModes(str, Enum):
    ABORT = auto()
    IGNORE = auto()
    PURGE = auto()


def _test_maze(maze_str):
    bd = Maze.bd_from_string(maze_str)
    bd.seed += 10
    return Maze.bd_to_string(bd)


def _initial_mazes(seed):
    train_bd = Maze.bd_from_string("5x5_U")
    train_bd.seed = 1000*seed
    train_bd_str = Maze.bd_to_string(train_bd)
    return [[train_bd_str], [_test_maze(train_bd_str)]]


class EDHuCATViewer(QDialog):
    class ScaledPixmapLabel(QLabel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image, self.scaled_image = None, None

        def set_image(self, image):
            self.image = image if isinstance(image, QImage) else QImage(image)
            self._scale_image()
            self.update()

        def _scale_image(self):
            if not self.image:
                return
            m = self.contentsMargins()
            m = 2 * max(m.right(), m.top(), m.left(), m.bottom())
            wr = (self.width() - m) / self.image.width()
            hr = (self.height() - m) / self.image.height()
            r = min(wr, hr)
            self.scaled_image = self.image.scaled(
                int(self.image.width() * r), int(self.image.height() * r),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if not self.isEnabled():
                self.scaled_image = self.scaled_image.convertToFormat(
                    QImage.Format_Grayscale8)

        def setEnabled(self, b: bool):
            super().setEnabled(b)
            self._scale_image()

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self._scale_image()

        def paintEvent(self, e):
            super().paintEvent(e)
            if self.scaled_image:
                painter = QPainter(self)
                img = self.scaled_image
                pos = QPointF(
                    .5 * (self.width() - img.width()),
                    .5 * (self.height() - img.height())
                )
                flags = Qt.ImageConversionFlags()
                flags |= Qt.MonoOnly
                painter.drawImage(pos, img, QRectF(img.rect()), flags)
                painter.end()

    def __init__(self, rows, base_folder, max_stages, folder_name_fn):
        super().__init__()
        self.name = "EDHuCAT Viewer"
        self.stage, self.max_stages = None, max_stages
        self.rows = rows
        self.base_folder = base_folder
        self.folder_name_fn = folder_name_fn

        self.setWindowFlags(
            Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)

        outer_layout = QGridLayout()
        # holder.setLayout(holder)
        self.setLayout(outer_layout)

        left_layout, right_layout = QVBoxLayout(), QGridLayout()
        outer_layout.addLayout(left_layout, 0, 0)
        outer_layout.addLayout(right_layout, 0, 1)

        right_layout.setSpacing(10)

        def field(name, obj):
            l_ = QHBoxLayout()
            l_.addWidget(QLabel(name), 0, Qt.AlignRight)
            if isinstance(obj, QWidget):
                l_.addWidget(obj, 1, Qt.AlignLeft)
            else:
                l_.addLayout(obj, 1)
            return l_

        self.showers: List[Tuple[QWidget, QLabel, QLabel, QPushButton]] = []
        self.controls = []
        self.train_makers, self.test_makers = [], []
        self.group = QButtonGroup()
        for i in range(rows):
            shower = QWidget()
            shower.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            lbl_cls = self.ScaledPixmapLabel
            shower_items = (shower, lbl_cls(), lbl_cls(), QPushButton())
            shower_layout = QHBoxLayout()
            for item in shower_items[1:]:
                shower_layout.addWidget(item)
            shower_items[-1].setSizePolicy(QSizePolicy.Fixed,
                                           QSizePolicy.Fixed)
            shower.setLayout(shower_layout)

            shower_items[-1].setCheckable(True)
            shower_items[-1].setChecked(False)
            shower_items[-1].toggled.connect(lambda t, _i=i:
                                             self._select_agent(_i, t))

            self.group.addButton(shower_items[-1], i)
            self.showers.append(shower_items)
            left_layout.addWidget(shower)

            train_maker = MainWindow(runnable=False)
            train_maker.maze_w.update_config(dark=True)
            right_layout.addWidget(train_maker.section_header("Train"),
                                   2*i, 0)
            right_layout.addWidget(train_maker.maze_w, 2*i+1, 0)

            test_maker = MainWindow(runnable=False)
            train_maker.maze_changed.connect(lambda _i=i: self._maze_changed(_i))
            test_maker.maze_w.update_config(dark=True)
            right_layout.addWidget(train_maker.section_header("Test"),
                                   2*i, 1)
            right_layout.addWidget(test_maker.maze_w, 2*i+1, 1)

            rightmost_layout = QVBoxLayout()
            rightmost_layout.setContentsMargins(0, 0, 0, 0)
            c_holder = QWidget()
            m_c_layout = QGridLayout()
            m_c_layout.setContentsMargins(0, 0, 0, 0)
            c_holder.setLayout(m_c_layout)
            right_layout.addLayout(rightmost_layout, 2*i, 2, 2, 1)
            rightmost_layout.addWidget(c_holder)
            r, c, cols = 0, 0, 3
            m_c_layout.addWidget(train_maker.section_header("Config"),
                                 r, 0, 1, cols)
            r += 1

            m_c_layout.addWidget(QLabel("Size"), r, c)
            s_layout = QHBoxLayout()
            s_layout.addWidget(train_maker.config["width"])
            s_layout.addWidget(QLabel("x"))
            s_layout.addWidget(train_maker.config["height"])
            m_c_layout.addLayout(field("Size", s_layout), r, 0)
            m_c_layout.addLayout(field("Seed", train_maker.config["seed"]), r, 1)
            train_maker.config["seed"].setRange(0, 9_999_999)
            m_c_layout.addWidget(train_maker.config["unicursive"], r, 2)
            r += 1
            for c, k in enumerate(["clues", "lures", "traps"]):
                m_c_layout.addWidget(train_maker.config["with_" + k], r, c)
                c += 2
            r += 1

            for maker, name in [(train_maker, "Train"), (test_maker, "Test")]:
                m_c_layout.addWidget(
                    maker.section_header(f"{name} maze"), r, 0, 1, cols)

                r += 1
                c = 0
                for c, k in enumerate(["m_size", "m_path", "m_intersections",
                                       "m_lures", "m_traps", "m_complexity"]):
                    name = k[2].upper() + k[3:].lower() + ":"
                    c_ = c % cols
                    r_ = r + c // cols
                    m_c_layout.addLayout(field(name, maker.stats[k]), r_, c_)

                r += math.ceil(c / cols)

            m_c_layout.addWidget(train_maker.section_header(""), r+1, 0, 1, 3)

            m_b_layout = QHBoxLayout()
            lbl = QLabel()
            lbl.setStyleSheet("background: green")
            lbl.setVisible(False)

            clone = QPushButton("Clone")
            clone.setFocusPolicy(Qt.NoFocus)
            clone.clicked.connect(lambda _, _i=i: self._duplicate_maze(_i))

            ok = QPushButton("Validate")
            ok.setCheckable(True)
            ok.setFocusPolicy(Qt.NoFocus)

            m_b_layout.addWidget(lbl)
            m_b_layout.addWidget(clone)
            m_b_layout.addStretch(1)
            m_b_layout.addWidget(ok)

            rightmost_layout.addLayout(m_b_layout)
            ok.toggled.connect(lambda _c, _i=i: self._validate_maze(_i, _c))

            rightmost_layout.addStretch(1)

            for j, s in enumerate([1, 1, 0]):
                right_layout.setColumnStretch(j, s)
            right_layout.setRowStretch(2*i, 0)
            right_layout.setRowStretch(2*i+1, 1)

            self.controls.append((c_holder, ok, clone, lbl))
            self.train_makers.append(train_maker)
            self.test_makers.append(test_maker)

        self.stats = QLabel()
        self.stats.setFont(QFont("Mono", 10))
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        decoy_layout = QHBoxLayout()
        decoy_layout.addStretch(1)
        decoy_layout.addWidget(self.stats)
        decoy_layout.addStretch(1)
        left_layout.addLayout(decoy_layout)

        feh = QPushButton("feh")
        feh.clicked.connect(self._to_feh)
        outer_layout.addWidget(feh, 1, 0, Qt.AlignLeft)

        self.validate = QPushButton("Proceed")
        self.validate.clicked.connect(self.accept)
        outer_layout.addWidget(self.validate, 1, 1, Qt.AlignRight)
        self.validate.setEnabled(False)

        self.group.idToggled.connect(self._process_selection)

        self.prev_mazes = None
        self.selected_id = None

        self.rng = random.Random(0) # TODO REMOVE

    def final_select_and_generate(self, stats):
        self.select_and_generate(self.stage+1, None, stats)

    def select_and_generate(self, stage, mazes, stats):
        self.stage = stage
        self.prev_mazes = mazes
        self.selected_id = None

        self.group.setExclusive(False)
        for a, s in enumerate(self.showers):
            trajectories = self._trajectory_files(stage-1, a)
            show = True
            for i, t in enumerate(trajectories):
                show &= t.exists()
                lbl: EDHuCATViewer.ScaledPixmapLabel = s[i+1]
                lbl.setVisible(show)
                if show:
                    lbl.set_image(str(t))
            s[0].setVisible(show)

            self._validate_maze(a, False)
            for c in self.controls[a][:-1]:
                c.setEnabled(False)
        self._select_agent(None)
        self.group.setExclusive(True)

        title = f"{self.name} - "
        if mazes:
            if len(mazes) == 1:
                self.showers[0][-1].click()
            title += f"Stage {stage} / {self.max_stages}"

        else:
            title += "Final stage"
            self._set_layout_visible(
                self.layout().takeAt(1).layout(), False)
        self.setWindowTitle(title)

        self.stats.setText(string_summary(stats))

        # self.move(0, 0)
        self.showMaximized()

        if False:  # TODO REMOVE
            def click():
                if stage == 1:
                    i = 0
                else:
                    i = self.rng.randint(0, self.rows-1)
                print(f"random {i=}")
                self.showers[i][-1].click()

                for _, b, _, _ in self.controls:
                    b.toggle()
                QTimer.singleShot(500, lambda: self.validate.click())

            d = 0#2000
            if d > 0:
                QTimer.singleShot(d, click)
            else:
                click()

        return self.exec()

    @staticmethod
    def _set_layout_visible(layout: QLayout, visible):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if w := item.widget():
                w.setVisible(visible)
            elif sl := item.layout():
                EDHuCATViewer._set_layout_visible(sl, visible)

    def _to_feh(self):
        args = ["feh", "-Z.", "--force-aliasing"]
        args.extend([str(f)
                     for a in range(self.rows)
                     for f in self._trajectory_files(self.stage-1, a)])
        subprocess.Popen(args)

    def _trajectory_files(self, stage, a):
        return [(self.base_folder
                 .joinpath(self.folder_name_fn(stage, a))
                 .joinpath(f"trajectories/{key}_final.png"))
                for key in ["train", "eval"]]

    def _process_selection(self, index, checked=None):
        if not checked:
            return

        self.selected_id = index
        if not self.prev_mazes:
            self.accept()
            return

        base_bd = Maze.bd_from_string(self.prev_mazes[index])
        for a, m in enumerate(self.train_makers):
            m.set_maze(copy.deepcopy(base_bd))
            for c in self.controls[a][:-1]:
                c.setEnabled(True)

    def _duplicate_maze(self, index):
        bd = self._maze_name(index)
        maze = Maze.bd_from_string(bd)
        for i, m in enumerate(self.train_makers):
            if i != index:
                m.set_maze(maze)
                self._validate_maze(i, False)

    def _select_agent(self, index=None, selected=None):
        def _process_one(i_, on):
            items = self.showers[i_]
            for label in items[1:-1]:
                label.setEnabled((on is None) or on)
            b: QPushButton = items[-1]
            _ = QSignalBlocker(b)
            b.setChecked(on or False)
            b.setIcon(QIcon.fromTheme("system-lock-screen"
                                      if on else "go-next"))

        if index is None:
            for i in range(len(self.showers)):
                _process_one(i, None)
        else:
            for i in range(len(self.showers)):
                if i != index:
                    _process_one(i, False)
            _process_one(index, selected)

    def _validate_maze(self, index, validated):
        group, button, clone, label = self.controls[index]
        group.setEnabled(not validated)
        button.setChecked(validated)
        button.setText("Cancel" if validated else "Validate")
        clone.setVisible(not validated)
        label.setText("" if not validated else self._maze_names(index))
        label.setVisible(validated)
        self.validate.setEnabled(all(c[1].isChecked() for c in self.controls))

    def _maze_name(self, index, makers=None):
        makers = makers or self.train_makers
        return Maze.bd_to_string(makers[index].maze_data())

    def _maze_names(self, index):
        return (self._maze_name(index)
                + "; "
                + self._maze_name(index, self.test_makers))

    def _maze_changed(self, index):
        bd = self._maze_name(index)
        bd_test = _test_maze(bd)
        self.test_makers[index].set_maze(Maze.bd_from_string(bd_test))

    def returns(self):
        winner = self.selected_id
        mazes = [self._maze_name(i) for i in range(self.rows)]
        return winner, mazes


def pretty_timeline(choices, train_data, base_folder: Path, prefix_fn):
    stages = len(choices)
    alternatives = len(choices[1][1])
    files = []
    for s, (a, _, _) in enumerate(choices):
        f_ = []
        for a_ in range(alternatives):
            path = (base_folder.joinpath(prefix_fn(s, a_))
                    .joinpath("trajectories/eval_final.png"))
            if path.exists():
                f_.append(path)
        files.append((a, f_))

    size, hm, vm = 512, 64, 8
    big_image = QImage((size + hm) * stages,
                       size * alternatives + vm * (alternatives - 1),
                       QImage.Format_RGB32)
    big_image.fill(Qt.white)
    painter = QPainter(big_image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    lines = []
    def _x(i): return (size + hm) * i
    def _y(j): return (size + vm) * j
    for s, (selection, slist) in enumerate(files):
        x = _x(s)
        for a, file in enumerate(slist):
            img = QImage(str(file))
            if len(slist) == 1:
                a = alternatives // 2
                selection = a
            painter.drawImage(QRectF(x, _y(a), size, size), img)
        lines.append((x, _y(selection) + .5*size))
    lines.append((_x(stages), _y(alternatives//2) + .5*size))

    train_data: pd.DataFrame
    timesteps: list = (
        train_data.loc[(slice(None), slice(None), slice("Train")), :]
        .unstack(0)['Steps'].values.transpose().tolist())
    consumptions = [lst[i] for lst, (i, _, _) in zip(timesteps, choices)]

    fm = painter.fontMetrics()
    for p0, p1, c in zip(lines[:-1], lines[1:], consumptions):
        x0, y0 = p0[0] + size, p0[1]
        x1, y1 = p1
        x_m, y_m = .5 * (x0 + x1), .5 * (y0 + y1)
        painter.drawLine(QLineF(x0, y0, x_m, y0))
        painter.drawLine(QLineF(x_m, y0, x_m, y1))
        painter.drawLine(QLineF(x_m, y1, x1, y1))

        cs = ""
        if c > 1_000_000:
            cs = "M"
            c //= 1_000_000
        elif c > 1_000:
            cs = "K"
            c //= 1_000
        text = f"{c}{cs}"
        rect = QRectF(fm.boundingRect(text))
        rect.moveCenter(QPointF(x_m, y_m))
        painter.fillRect(rect, Qt.white)
        painter.drawText(rect, Qt.AlignCenter, text)

    painter.end()
    big_image.save(str(base_folder.joinpath("timeline.png")))
    args = ["feh", "-Z.", str(base_folder.joinpath("timeline.png"))]
    subprocess.Popen(args)


def store_train_data(data: Optional[pd.DataFrame], i, j, items):
    if data is None:
        data = pd.DataFrame(
            index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                                names=['I', 'J', "T"])
        )

    keys = {
        "pretty_reward": "PR",
        "reward": "RR",
        **{f"errors/{t.value.lower()}": t.value[0].upper() for t in SignType}
    }

    for r in ["train", "eval"]:
        r_ = r[0].upper() + r[1:].lower()
        for k, name in keys.items():
            value = items[0][r + "/" + k]
            data.loc[(i, j, r_), name] = value

    data.loc[(i, j, "Train"), ["Steps", "Use"]] = items[1:]
    return data


def store_maze_stats(data: Optional[pd.DataFrame], i, mazes) -> pd.DataFrame:
    if data is None:
        data = pd.DataFrame(
            columns=["W", "H", "S", "C", "L", "T", "Emin", "Emax", "Name"],
            index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                                names=["I", "J", "T"]))

    def _set(m_str, j, k):
        m = Maze.from_string(m_str)
        c = Simulation.compute_complexity(m, InputType.DISCRETE, 15)
        data.loc[(i, j, k), :] = [
            m.width, m.height, m.seed,
            *[len(m.signs_data[s_type]) for s_type in SignType],
            *c["entropy"].values(), m_str]

    train_maze, eval_maze = mazes
    for j, (t, e) in enumerate(zip(train_maze, eval_maze)):
        _set(t, j, "Train")
        _set(e, j, "Eval")

    return data


def string_summary(stats):
    stats: list[pd.DataFrame]

    stats = [s.groupby(level=1).tail(2) for s in stats]

    index_strs = stats[0].to_string(
        formatters=[lambda _: "" for _ in range(len(stats[0].columns))],
        header=False).split("\n")
    whitespace = max([len(s.rstrip()) for s in index_strs])
    index_strs = [s[:whitespace] for s in index_strs]
    stats_strs = [stat.to_string(na_rep="", index=False,
                                 float_format=lambda v: f"{round(v, 2):g}")
                  .split('\n')
                  for stat in stats]

    def row(i_, l_, r_): return f"{i_}   {l_}   {r_}\n"

    widths = [len(s[0]) for s in [index_strs, *stats_strs]]
    text = row(' ' * widths[0],
               f"{'Mazes':^{widths[1]}s}",
               f"{'Training':^{widths[2]}s}")
    text += row(' ' * widths[0], '-' * widths[1], '-' * widths[2])
    for index, lhs, rhs in zip(*[s for s in [index_strs, *stats_strs]]):
        text += row(index, lhs, rhs)

    return text


def log_summary(stats: list[pd.DataFrame]):
    msg = f"\n{'='*80}\n== Stage summary:\n"
    msg += string_summary(stats)
    logger.info(msg)


TRAINERS = {
    (InputType.DISCRETE, OutputType.DISCRETE): [
        A2C, DQN, PPO,
    ],
    # (InputType.DISCRETE, OutputType.CONTINUOUS): [
    (InputType.CONTINUOUS, OutputType.DISCRETE): [
        A2C, DQN, PPO
    ],
    (InputType.CONTINUOUS, OutputType.CONTINUOUS): [
        A2C,      PPO, SAC, TD3
    ]
}


@dataclass
class Options:
    id: Optional[Union[str, int]] = None
    base_folder: Path = Path("tmp/sb3/edhucat")
    run_folder: Path = None  # automatically filled in
    overwrite: OverwriteModes = OverwriteModes.ABORT
    verbosity: int = 0
    quietness: int = 0

    trainer: str = None
    budget: int = 3_000_000
    stages: int = 10
    alternatives: int = 3
    trajectories: int = 10
    evals: int = 100

    seed: int = None

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument("--id", dest="id", metavar="ID",
                            help="Identifier for the run (and seed if integer"
                                 "and seed is not provided)")

        parser.add_argument("-f", "--folder", dest="base_folder",
                            metavar="F", type=Path,
                            help="Base folder under which to store data")

        parser.add_argument("--overwrite", dest="overwrite",
                            choices=[o.name for o in OverwriteModes],
                            type=str.upper, metavar="O", nargs='?',
                            help="Purge target folder before started")

        parser.add_argument("-v", "--verbose", dest="verbosity",
                            action='count', help="Increase verbosity level")

        parser.add_argument("-q", "--quiet", dest="quietness",
                            action='count', help="Decrease verbosity level")

        parser.add_argument("--trainer", dest="trainer",
                            choices=list(set(v.__name__
                                             for vl in TRAINERS.values()
                                             for v in vl)),
                            type=str.upper, metavar="T",
                            required=True,
                            help="What RL algorithm to use")

        parser.add_argument("--budget", dest="budget", metavar="B",
                            type=int, required=True,
                            help="Total number of timesteps")

        parser.add_argument("--stages", dest="stages", metavar="N",
                            type=int, required=True,
                            help="Number of environment used")

        parser.add_argument("--lambda", dest="alternatives",
                            metavar="A", type=int,  required=True,
                            help="Number of concurrent training")

        parser.add_argument("--trajectories", metavar='N', type=int,
                            help="Number of trajectories to log")

        parser.add_argument("--evals", metavar='E', type=int,
                            help="Number of intermediate evaluations")

        group = parser.add_argument_group(
            "Reinforcement", "Settings for all reinforcement learning types")
        group.add_argument("--seed", dest="seed", help="Seed for RNG",
                           type=int, metavar="S")

    def normalize(self):
        self.verbosity -= self.quietness

        # Generate id if needed
        id_needed = (self.id is None)
        if id_needed:
            if self.seed is None:
                self.id = time.strftime('%m%d%H%M%S')
            else:
                self.id = self.seed

        # Define the run folder
        folder_name = self.id
        if not isinstance(folder_name, str):
            folder_name = f"run{self.id}"
        self.run_folder = self.base_folder.joinpath(folder_name).resolve()

        if self.run_folder.exists():
            if self.overwrite is None:
                self.overwrite = OverwriteModes.PURGE
            if not isinstance(self.overwrite, OverwriteModes):
                self.overwrite = OverwriteModes[self.overwrite]
            if self.overwrite is OverwriteModes.IGNORE:
                logger.info(f"Ignoring contents of {self.run_folder}")

            elif self.overwrite is OverwriteModes.PURGE:
                shutil.rmtree(self.run_folder, ignore_errors=True)
                logger.warning(f"Purging contents of {self.run_folder},"
                               f" as requested")

            else:
                raise OSError("Target directory exists. Aborting")
        self.run_folder.mkdir(parents=True, exist_ok=True)

        tee = Tee(self.run_folder.joinpath("log"))

        log_level = 20-self.verbosity*10
        logging.basicConfig(level=log_level, force=True,
                            stream=sys.stdout,
                            format="[%(asctime)s|%(levelname)s|%(module)s]"
                                   " %(message)s",
                            datefmt="%Y-%m-%d|%H:%M:%S")
        logging.addLevelName(60, "KGD")
        logging.addLevelName(logging.WARNING, "WARN")
        logger.log(
            60, f"Using log level of {logging.getLevelName(logger.root.level)}")
        for m in ['matplotlib']:
            logger_ = logging.getLogger(m)
            logger_.setLevel(logging.WARNING)
            logger.info(f"Muting {logger_}")
        logger.setLevel(log_level-10)

        if id_needed:
            logger.info(f"Generated run id: {self.id}")
        logger.info(f"Run folder: {self.run_folder}")

        if self.seed is None:
            try:
                self.seed = int(self.id)
            except ValueError:
                self.seed = round(1000 * time.time())
            logger.info(f"Deduced seed: {self.seed}")

        # # Check the thread parameter
        # options.threads = max(1, min(options.threads, len(os.sched_getaffinity(0))))
        # logger.info(f"Parallel: {options.threads}")

        if self.verbosity >= 0:
            raw_dict = {f.name: getattr(self, f.name) for f in fields(self)}
            logger.info(f"Post-processed command line arguments:"
                        f"\n{pprint.pformat(raw_dict)}")

        return log_level, tee


def agg(e, f_, f__): return f_(e.env_method(f__))


def make_envs(maze, robot, seed, log_trajectory=False):
    data = maze
    if isinstance(data, Maze.BuildData):
        data = maze
    elif isinstance(data, str):
        data = Maze.bd_from_string(maze)
    else:
        raise ValueError(f"Wrong argument type {type(maze)} instead of"
                         f" Maze.BuildData")
    mazes = data.all_permutations()

    def _make_env(env_list: list):
        env = MazeEnv(maze=env_list.pop(0), robot=robot,
                      log_trajectory=log_trajectory)
        check_env(env)
        env.reset(full_reset=True)
        return env

    return make_vec_env(
        functools.partial(_make_env, env_list=mazes),
        n_envs=len(mazes), seed=seed,
        vec_env_cls=DummyVecEnv)


def _run_one(args, budget, prefix, env_fn, model_fn, train_maze, eval_maze):
    pretty_prefix = prefix.replace('_', ' ')
    pretty_prefix = pretty_prefix[0].upper() + pretty_prefix[1:].lower()

    stage_folder_path = args.run_folder.joinpath(prefix)
    stage_folder = str(stage_folder_path)

    train_env = env_fn(train_maze)
    eval_env = env_fn(eval_maze, log_trajectory=True)
    n_train_envs, n_eval_envs = train_env.num_envs, eval_env.num_envs

    model = model_fn(train_env)

    # Periodically evaluate agent
    trajectories_freq = args.trajectories
    train_duration = agg(train_env, np.sum, "maximal_duration")
    if args.evals == -1:
        eval_freq = train_duration
        logger.debug(f"Evaluating after every full training"
                     f" ({eval_freq} timesteps)")
    else:
        eval_freq = budget // args.evals
        if eval_freq < train_duration:
            eval_freq = train_duration
            logger.warning(f"Not enough budget for {args.evals} evaluations:\n"
                           f" train duration={train_duration}"
                           f" budget={budget}, evals={args.evals};"
                           f" {args.evals} > {budget / train_duration}\n"
                           f" Clamping to"
                           f" {int(budget // train_duration)}")
        else:
            logger.debug(f"Evaluating every {eval_freq} call to env.step()"
                         f" (across {n_train_envs} evaluation environments)")
    optimal_reward = agg(eval_env, np.average, "optimal_reward")
    logger.debug(f"Training will stop upon reaching average reward of"
                 f" {optimal_reward}")
    tb_callback = TensorboardCallback(
        log_trajectory_every=trajectories_freq,
        max_timestep=args.budget,
        prefix=prefix, multi_env=True
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None, log_path=stage_folder,
        eval_freq=max(1, eval_freq // n_eval_envs),
        n_eval_episodes=n_eval_envs,
        deterministic=True, render=False, verbose=args.verbosity,
        callback_after_eval=tb_callback,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=optimal_reward,
            verbose=1)
    )

    model.set_env(train_env)

    # Store the policy type for agnostic reload
    setattr(model, "model_class", model.__class__)
    with open(args.run_folder.joinpath("best_model.class"), 'wt') as f:
        f.write(model.__class__.__name__ + "\n")

    model.set_logger(configure(stage_folder,
                               ["csv", "tensorboard"]))
    start_timesteps = model.num_timesteps

    # logger.info("[kgd-debug] Before manual eval callback")
    # eval_callback.init_callback(model)
    # eval_callback.num_timesteps = model.num_timesteps
    # eval_callback._on_step()
    # tb_callback.log_step(False)
    # logger.info("[kgd-debug] After manual eval callback")

    model.learn(budget, callback=eval_callback, progress_bar=True,
                reset_num_timesteps=False)

    logger.debug(f"[{pretty_prefix}] Performing final logging step manually")
    tb_callback.log_step(True)

    logger.info(f"[{pretty_prefix}] End of stage:"
                f" {start_timesteps} -> {model.num_timesteps}")

    save_path = stage_folder_path.joinpath("best_model.zip")
    logger.debug(f"[{prefix}] Saving model to {save_path}")
    model.save(save_path)

    consumed = model.num_timesteps - start_timesteps
    if consumed > budget:
        logger.warning(f"[{pretty_prefix}] Overconsumption:"
                       f" {consumed} > {budget}")

    return model, save_path, tb_callback.last_stats, consumed


def main():
    with CV2QTGuard(platform=False):
        _ = QApplication([])

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    start = time.perf_counter()

    args = Options()
    parser = argparse.ArgumentParser(
        description="Main trainer for maze EDHuCATion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n------------------\n"
               "Additional options\n"
    )
    Options.populate(parser)
    parser.parse_args(namespace=args)
    # noinspection PyUnusedLocal
    logging_level, tee = args.normalize()

    robot = Robot.BuildData.from_argparse(args)
    logger.info(f"Using\n{pprint.pformat(robot)}")

    _make_envs = functools.partial(make_envs, robot=robot, seed=args.seed)

    trainer = next((t for t in TRAINERS[(robot.inputs, robot.outputs)]
                    if t.__name__.casefold() == args.trainer.casefold()),
                   None)
    if not trainer:
        raise ValueError(f"Requested trainer {args.trainer} is not valid"
                         f" for given {robot.inputs}->{robot.outputs}"
                         f" combination")
    policy = {
        InputType.DISCRETE: "MlpPolicy",
        InputType.CONTINUOUS: "CnnPolicy"
    }[robot.inputs]

    logger.info(f"Training with {trainer.__name__} supported by {policy}")
    main_folder = args.run_folder.joinpath("timeline")
    main_folder.mkdir(parents=True, exist_ok=False)

    def symlink(target, _name=None):
        print(target, _name)
        if not _name:
            _name = target.name
        _symlink = main_folder.joinpath(_name)
        target = Path("../").joinpath(target.relative_to(args.run_folder))
        logger.info(f"{_symlink} -> {target}")
        _symlink.symlink_to(target)
        assert _symlink.resolve().exists(), f"{_symlink.resolve()}"

    def symlink_save(model_path: Path, prefix_: str):
        symlink(model_path, _name=f"best_model_{prefix_}.zip")
        if events := list(model_path.parent.glob("*tfevents*")):
            if len(events) != 1:
                logger.warning(f"Found more than one event file under"
                               f" {model_path.parent}")
            symlink(events[0])
        for trajectory in model_path.parent.glob("trajectories/*.png"):
            symlink(trajectory, _name=f"traj_{prefix_}_{trajectory.name}")

    _stage_prefix = (
        f"stage"
        f"_{{:0{math.ceil(math.log10(args.stages))}}}"
        f"_{{:0{math.ceil(math.log10(args.alternatives))}}}")

    def stage_prefix(s_, a_): return _stage_prefix.format(s_, a_)

    remaining_budget = args.budget

    last_model_path: Optional[Path] = None

    def _make_or_load_model(env):
        if last_model_path is None:
            logger.info(f"[{prefix}] Generating new model")
            return trainer(policy, env, seed=args.seed, learning_rate=1e-3)
        else:
            logger.info(f"[{prefix}] Loading model from"
                        f" {last_model_path}")
            return trainer.load(last_model_path)

    logger.info(f"\n{'=' * 35} Starting {'=' * 35}")

    def _run(**kwargs):
        return _run_one(args=args, env_fn=_make_envs,
                        model_fn=_make_or_load_model, **kwargs)

    prefix = stage_prefix(0, 0)
    budget = args.budget / args.stages
    train_mazes, eval_mazes = _initial_mazes(args.seed)
    logger.info(f"\n{'=' * 80}\n== Stage {0}\n{train_mazes=}\n{eval_mazes=}")
    m, p, r, t = _run(budget=budget, prefix=prefix,
                      train_maze=train_mazes[0], eval_maze=eval_mazes[0])
    remaining_budget -= t

    with CV2QTGuard(platform=False):
        dialog = EDHuCATViewer(args.alternatives, args.run_folder,
                               args.stages, stage_prefix)

    choices = []

    train_data = store_train_data(None, 0, 0, [r, t, 100*t/budget])
    mazes_data = store_maze_stats(None, 0, [train_mazes, eval_mazes])
    log_summary([mazes_data, train_data])

    models = [p]

    for stage in range(1, args.stages):
        with CV2QTGuard(platform=False):
            dialog.select_and_generate(stage, train_mazes,
                                       [mazes_data, train_data])
            selection, mazes = dialog.returns()
        logger.info(f"\n{'='*80}\n== Stage {stage} {selection=} {mazes=}")

        last_model_path = models[selection]
        symlink_save(last_model_path, stage_prefix(stage-1, selection))
        choices.append([selection, train_mazes, eval_mazes])

        train_mazes = mazes
        eval_mazes = [_test_maze(m) for m in train_mazes]

        models = []
        budget = math.floor(remaining_budget /
                            ((args.stages - stage) * args.alternatives))
        for a, (m_train, m_eval) in enumerate(zip(train_mazes, eval_mazes)):
            prefix = stage_prefix(stage, a)

            m, p, r, t = (
                _run(budget=budget, prefix=prefix,
                     train_maze=m_train, eval_maze=m_eval))
            models.append(p)

            remaining_budget -= t
            store_train_data(train_data, stage, a, [r, t, 100*t/budget])

        store_maze_stats(mazes_data, stage, [train_mazes, eval_mazes])
        log_summary([mazes_data, train_data])

    for df, name in [(mazes_data, "mazes"), (train_data, "training")]:
        df.to_csv(main_folder.joinpath(name + ".csv"),
                  sep=' ', float_format="{:g}")

    with CV2QTGuard(platform=False):
        dialog.final_select_and_generate([mazes_data, train_data])
        selection = dialog.selected_id
    choices.append([selection, [], []])
    logger.info(f"Final stage {selection=}")

    last_model_path = models[selection]
    symlink_save(last_model_path, "stage_final")

    with CV2QTGuard():
        pretty_timeline(choices, train_data, args.run_folder, stage_prefix)

    total_consumption = args.budget - remaining_budget

    msg = ""
    if remaining_budget > 0:
        msg += (f"Training converged in {total_consumption}"
                f" / {args.budget} time steps")
    elif remaining_budget == 0:
        msg += f"Training completed in {args.budget}"
    else:
        msg += (f"Training completed in {args.budget}"
                f" \033[31m+{-remaining_budget}\033[0m")
    logger.info(f"{msg}.")

    # =========================================================================

    duration = humanize.precisedelta(
        timedelta(seconds=time.perf_counter() - start))
    logger.info(f"Completed training in {duration}")
    logger.info(f"Results are under {args.run_folder}")


if __name__ == "__main__":
    #
    # for seed in ["M0_", ""]:
    #     for specs in ["U", "C1", "C1_T.5",
    #                   "C1_C.5_T.25_T.75", "C1_C.75_T.25_T.5"]:
    #         maze = seed + "5x5_" + specs
    #         r = Simulation.compute_complexity(
    #             Maze.from_string(maze), InputType.DISCRETE, 15)
    #         print(f"{maze}:\n> {pprint.pformat(r)}")
    #
    # exit(42)

    main()
