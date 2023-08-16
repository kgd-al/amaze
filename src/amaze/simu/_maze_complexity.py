import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QApplication

from amaze.simu.env.maze import Maze
from amaze.simu.robot import InputType
from amaze.visu.widgets.labels import InputsLabel


def __inputs(c, r, maze, visuals):
    i = np.zeros(8)
    i[:4] = [maze.wall(c, r, d) for d in Maze.Direction]

    if d := visuals[r, c]:
        if not isinstance(d, float) or not np.isnan(d):
            i[4 + d[1].value] = d[0]

    return i


def __all_inputs(maze: Maze, visuals):
    inputs = defaultdict(lambda: 0)
    cells = maze.iter_cells if not maze.unicursive() else maze.iter_solutions

    i = __inputs(*maze.start, maze=maze, visuals=visuals)
    inputs[tuple(i)] += 1

    for c, r in cells():
        i = __inputs(r=r, c=c, maze=maze, visuals=visuals)

        for j in range(4):
            if not i[j]:
                i_ = i.copy()
                i_[j] = .5
                inputs[tuple(i_)] += 1

    return inputs


def __solution_path(maze, visuals):
    inputs = defaultdict(lambda: 0)
    for j, (c, r) in enumerate(maze.solution):
        i = __inputs(r=r, c=c, maze=maze, visuals=visuals)
        if j > 0:
            c0_i, c0_j = maze.solution[j - 1]
            c1_i, c1_j = maze.solution[j]
            di, dj = c0_i - c1_i, c0_j - c1_j
            d = maze._offsets_inv[(di, dj)]
            i[d.value] = .5
        inputs[tuple(i)] += 1
    return inputs


def __entropy_bits(inputs):
    entropy = 0
    total_states = sum(inputs.values())
    for n in inputs.values():
        p = n / total_states
        entropy += - p * math.log(p)
    return entropy


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


def complexity(maze: Maze, visuals: np.ndarray):

    inputs = []
    entropy = []
    for f, k in [(__solution_path, "minimal"),
                 (__all_inputs, "maximal")]:
        i = f(maze, visuals)
        inputs.append(len(i))
        entropy.append(__entropy_bits(i))

        __debug_draw_inputs(i, maze.to_string())

    # print(f"{inputs=} {entropy=}")
    return inputs, entropy
