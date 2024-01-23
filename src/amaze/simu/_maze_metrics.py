import enum
import math
import pprint
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Callable, Tuple, Generator

import numpy as np
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QApplication

from amaze.simu.env.maze import Maze
from amaze.simu.robot import InputType
from amaze.visu.resources import Sign
from amaze.visu.widgets.labels import InputsLabel


class MazeMetrics(enum.Flag):
    SURPRISINGNESS = enum.auto()
    DECEPTIVENESS = enum.auto()
    INSEPARABILITY = enum.auto()
    ALL = SURPRISINGNESS | DECEPTIVENESS | INSEPARABILITY


def __inputs(c, r, maze, visuals):
    i = np.zeros(8)
    i[:4] = [maze.wall(c, r, d) for d in Maze.Direction]

    if d := visuals[c, r]:
        if not isinstance(d, float) or not np.isnan(d):
            i[4 + d[1].value] = d[0]

    return i


def __all_inputs(maze: Maze, visuals):
    inputs = defaultdict(lambda: 0)
    cells = maze.iter_cells if not maze.unicursive() else maze.iter_solutions

    i = __inputs(*maze.start, maze=maze, visuals=visuals)
    yield tuple(i)

    for c, r in cells():
        i = __inputs(c=c, r=r, maze=maze, visuals=visuals)

        for j in range(4):
            if not i[j]:
                i_ = i.copy()
                i_[j] = .5
                yield tuple(i_)


def __solution_path(maze, visuals):
    inputs = defaultdict(lambda: 0)
    for j, (c, r) in enumerate(maze.solution):
        i = __inputs(c=c, r=r, maze=maze, visuals=visuals)
        if j > 0:
            c0_i, c0_j = maze.solution[j - 1]
            c1_i, c1_j = maze.solution[j]
            di, dj = c0_i - c1_i, c0_j - c1_j
            d_from = maze._offsets_inv[(di, dj)]
            i[d_from.value] = .5

        d_to = None
        if j < len(maze.solution) - 1:
            c1_i, c1_j = maze.solution[j]
            c2_i, c2_j = maze.solution[j + 1]
            di, dj = c1_i - c2_i, c1_j - c2_j
            d_to = maze._offsets_inv[(di, dj)]

        yield tuple(i), d_to


class InputsEntropy:
    def __init__(self):
        self.__count = 0
        self.__inputs = defaultdict(lambda: 0)

    def process(self, state: Tuple):
        self.__count += 1
        self.__inputs[hash(state)] += 1

    def value(self):
        entropy = 0
        for n in self.__inputs.values():
            p = n / self.__count
            entropy += - p * math.log(p)
        return entropy

    def count(self): return self.__count
    def reset(self): self.__init__()


class StatesEntropy:
    def __init__(self):
        self.__counts = defaultdict(lambda: 0)
        self.__inputs = defaultdict(lambda: defaultdict(lambda: 0))

    def process(self, state: Tuple, solution: Maze.Direction):
        h = hash(tuple([*state[:4], solution]))
        self.__counts[h] += 1
        self.__inputs[h][hash(state[4:])] += 1

    def value(self):
        entropy = 0
        for cell, states in self.__inputs.items():
            for n in states.values():
                p = n / self.__counts[cell]
                entropy += - p * math.log(p)
        return entropy


# noinspection PyPep8Naming
def __mutual_information(maze: Maze):
    c, t = maze.clues(), maze.traps()
    if len(c) + len(t) == 0:
        return 0

    TRUTH = {s.value: v for signs, v in [(c, True), (t, False)] for s in signs}
    X = set(TRUTH.keys())
    Y = set(TRUTH.values())
    P_XY = {(x.value, y): y == v for signs, v in [(c, True), (t, False)]
            for x in signs for y in Y}
    P_X = 1 / len(X)
    P_Y = {y: 0 for y in Y}

    X_ = sorted(list(X))
    for i, (s, v) in enumerate(sorted(TRUTH.items())):
        left = 0 if i == 0 else .5 * (s + X_[i-1])
        right = 1 if i == len(X)-1 else .5 * (X_[i+1] + s)
        P_Y[v] += right - left

    mi = 0
    for y in Y:
        for x in X:
            p_xy, mp = P_XY[(x, y)], P_X * P_Y[y]
            if mp and p_xy:
                mi += p_xy * math.log2(p_xy / mp)

    return mi


def __inseparability(maze: Maze):
    c, l, t = maze.clues(), maze.lures(), maze.traps()
    print(c, l, t)
    sorted_signs = sorted([(s.value, t) for lst, t in [(c, Sign.CLUE),
                                                       (l, Sign.LURE),
                                                       (t, Sign.TRAP)]
                           for s in lst], key=lambda x: x[0])
    print(sorted_signs)
    range_t = namedtuple("Range", ["l", "u", "t"])

    ranges = [range_t(0, *sorted_signs[0])]
    for v, t in sorted_signs[1:]:
        last = ranges[-1]
        if last.t == t:
            ranges[-1] = range_t(last.l, v, t)
        else:
            mid = .5 * (last.u + v)
            ranges[-1] = range_t(last.l, mid, last.t)
            ranges.append(range_t(mid, v, t))
    ranges[-1] = range_t(ranges[-1].l, 1, ranges[-1].t)
    print()
    print("=====")
    pprint.pprint(ranges)
    print("=====")
    print()
    return 0


def __debug_draw_inputs(inputs, name):
    app = QApplication([])
    label = InputsLabel()
    label.setFixedSize(5, 5)
    folder = Path(f"tmp/complexity/{name}")
    folder.mkdir(parents=True, exist_ok=True)
    for i in inputs:
        str_i = "_".join(f"{v:g}" for v in i)
        img = QImage(5, 5, QImage.Format_RGB32)
        painter = QPainter(img)
        label.set_inputs(np.array(i), InputType.DISCRETE)
        label.render(painter)
        painter.end()
        img.save(str(folder.joinpath(f"{str_i}.png")))


def metrics(maze: Maze, visuals: np.ndarray, input_type: InputType):
    if input_type is not InputType.DISCRETE:
        raise NotImplementedError

    n_inputs = {}
    s_entropy = {}

    # ======

    k = "all"
    s_metric = InputsEntropy()
    for i in __all_inputs(maze, visuals):
        s_metric.process(i)
    n_inputs[k] = s_metric.count()
    s_entropy[k] = s_metric.value()

    # ======

    k = "path"
    s_metric.reset()
    d_metric = StatesEntropy()
    for i, d in __solution_path(maze, visuals):
        s_metric.process(i)
        d_metric.process(i, d)

    n_inputs[k] = s_metric.count()
    s_entropy[k] = s_metric.value()
    d_entropy = d_metric.value()

    # ======

    # __debug_draw_inputs(i, maze.to_string())

    return {
        MazeMetrics.INSEPARABILITY: __inseparability(maze),
        MazeMetrics.SURPRISINGNESS: s_entropy,
        MazeMetrics.DECEPTIVENESS: d_entropy,

        "n_inputs": n_inputs,
        "mutual_information": __mutual_information(maze),
        #
        # "__debug_entropy": {
        #     k: __entropy_bits(f(maze, visuals)) for k, f in [("all", __all_inputs), ("path", __solution_path)]
        # }
    }
