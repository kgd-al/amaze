import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QColor, QPainter, QImage

from amaze.misc.resources import SignType, arrow_path, Sign, np_images, Signs
from .controllers.base import BaseController
from .maze import Maze
from .types import Action

ARROW_PATH = arrow_path()

DISCRETE_INPUTS_IMAGE_SIZE = 32
DISCRETE_INPUTS_WALL_SIZE = DISCRETE_INPUTS_IMAGE_SIZE // 16


def _image_inputs_image_size(vision: int):
    if vision * 2 >= DISCRETE_INPUTS_IMAGE_SIZE:
        return 1, vision
    else:
        scale = math.ceil(DISCRETE_INPUTS_IMAGE_SIZE / vision)
        return scale, scale * vision


Direction = Maze.Direction
OrientedSign = Tuple[Sign, SignType, Direction]
InputDetails = Tuple[
    np.ndarray,  # Walls
    Action,  # Correct action
    Optional[Direction],  # Previous direction
    Optional[OrientedSign],
]  # Sign and orientation


def __image(size):
    image = QImage(size, size, QImage.Format_RGB32)
    image.fill(Qt.black)
    painter = QPainter(image)
    painter.setPen(Qt.white)

    return image, painter


def __draw_arrow(painter, color, rotation, size, height):
    painter.save()
    painter.rotate(-90 * rotation)
    painter.scale(0.9 * size, 0.9 * height * size)
    painter.translate(-0.5, -0.5)
    painter.fillPath(ARROW_PATH, color)
    painter.restore()


def __draw_action(painter: QPainter, correct_action, action, size: float):
    def a_to_dir(a):
        return Maze.offsets_inv[a]

    correct_direction = a_to_dir(correct_action)
    if correct_action != action:
        __draw_arrow(painter, Qt.blue, correct_direction.value, size, 0.5)

    if action is not None:
        selected_direction = a_to_dir(action)
        if selected_direction == correct_direction:
            __draw_arrow(painter, Qt.green, correct_direction.value, size, 0.5)
        else:
            __draw_arrow(painter, Qt.red, selected_direction.value, size, 0.5)


def __draw_discrete_input(details: InputDetails, _, action: Action = None):
    size = DISCRETE_INPUTS_IMAGE_SIZE
    image, painter = __image(size)

    painter.scale(size, size)
    painter.translate(0.5, 0.5)

    w = DISCRETE_INPUTS_WALL_SIZE / size
    walls, correct_action, prev_dir, oriented_sign = details
    for i, v in enumerate(walls):
        if v == 1:
            painter.fillRect(QRectF(0.5 - w, -0.5, w, 1), QColor.fromHsvF(0, 0, v))
        elif prev_dir and i == prev_dir.value:
            painter.fillRect(QRectF(0.5 - w, -w / 2, w, w), Qt.red)
        painter.rotate(-90)

    if oriented_sign is not None:
        sign, _, direction = oriented_sign
        __draw_arrow(
            painter,
            QColor.fromHsvF(0, 0, sign.value),
            direction.value,
            1 - w,
            1,
        )

    __draw_action(painter, correct_action, action, 1 - w)

    painter.end()
    return image


def __draw_image_input(details: InputDetails, visual: np.ndarray, action: Action):

    scale, size = _image_inputs_image_size(visual.shape[1])
    image, painter = __image(size)

    w, h = visual.shape
    img = QImage(
        (visual * 255).astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1).data,
        scale * w,
        scale * h,
        scale * w,
        QImage.Format_Grayscale8,
    )
    painter.drawImage(0, 0, img)

    painter.scale(size, size)
    painter.translate(0.5, 0.5)
    _, correct_action, *_ = details
    __draw_action(painter, correct_action, action, 1 - 1 / (w - 2))

    painter.end()
    return image


def __draw_inputs(
    path: Path,
    inputs: List[InputDetails],
    visuals: List[np.ndarray],
    outputs: Optional[List[Action]],
    individual_files: bool,
    summary_file: bool,
    summary_file_ratio,
):
    i_type_list = {t: i for i, t in enumerate([None, SignType.LURE, SignType.CLUE, SignType.TRAP])}

    def sign_type(_i):
        return _s[1] if (_s := _i[3]) is not None else None

    inputs, visuals, outputs = zip(
        *sorted(
            itertools.zip_longest(inputs, visuals, outputs or []),
            key=lambda _i: i_type_list[sign_type(_i[0])],
        )
    )

    shape = visuals[0].shape
    if len(shape) == 1:
        drawer = __draw_discrete_input
        size = DISCRETE_INPUTS_IMAGE_SIZE

    else:
        drawer = __draw_image_input
        _, size = _image_inputs_image_size(shape[1])
        print(size)

    n_inputs = len(inputs)
    i_digits = math.ceil(math.log10(n_inputs))
    cols = math.ceil(math.sqrt(n_inputs) * summary_file_ratio)
    rows = math.ceil(n_inputs / cols)

    margin = 4
    if summary_file:
        big_image = QImage(cols * (size + margin), rows * (size + margin), QImage.Format_ARGB32)
        big_image.fill(Qt.transparent)
        painter = QPainter(big_image)

    folder = path.parent if path.is_file() else path
    for i, (obs, visual, action) in enumerate(
        itertools.zip_longest(inputs, visuals, outputs or [])
    ):
        img = drawer(obs, visual, action)
        if individual_files:
            img.save(str(folder.joinpath(f"{i:0{i_digits}d}.png")))

        if summary_file:
            x, y = i % cols, i // cols
            painter.drawImage(
                QPointF(
                    x * (size + margin) + 0.5 * margin,
                    y * (size + margin) + 0.5 * margin,
                ),
                img,
            )

    if summary_file:
        painter.end()
        big_image.save(str(path))


def _all_inputs(signs: dict[SignType, Signs]) -> List[InputDetails]:
    inputs: List[InputDetails] = []

    def array():
        return np.zeros(4)

    def i_to_dir(i_):
        return Maze.offsets[Maze.Direction(i_)]

    lures = signs[SignType.LURE]
    traps = signs[SignType.TRAP]
    clues = signs[SignType.CLUE]

    # First the dead ends
    for i in range(4):
        a = array()
        a[:4] = 1
        a[i] = 0
        for d in [None, Direction(i)]:
            a = a.copy()
            inputs.append((a, i_to_dir(i), d, None))

            for lure in lures:
                for lure_dir in iter(set(range(4)) - {i}):
                    a_ = a.copy()
                    inputs.append(
                        (
                            a_,
                            i_to_dir(i),
                            d,
                            (lure, SignType.LURE, Direction(lure_dir)),
                        )
                    )

    # Then the corridors
    for i in range(3):
        for j in range(i + 1, 4):
            dirs = set(range(4)) - {i, j}
            for d in dirs:
                a = array()
                a[i] = 1
                a[j] = 1
                i_sol = next(iter(dirs - {d}))
                sol = i_to_dir(i_sol)
                inputs.append((a, sol, Direction(d), None))
                for lure in lures:
                    for lure_dir in iter(set(range(4)) - {i_sol}):
                        a_ = a.copy()
                        inputs.append(
                            (
                                a_,
                                sol,
                                Direction(d),
                                (lure, SignType.LURE, Direction(lure_dir)),
                            )
                        )

    # And the intersections
    for i in range(4):
        for j in iter(set(range(4)) - {i}):
            for k in iter(set(range(4)) - {i}):
                a = array()
                a[i] = 1
                for clue in clues:
                    inputs.append(
                        (
                            a,
                            i_to_dir(j),
                            Direction(k),
                            (clue, SignType.CLUE, Direction(j)),
                        )
                    )

                if j != k:
                    a = a.copy()
                    for trap in traps:
                        i_sol = next(iter(set(range(4)) - {i, j, k}))
                        inputs.append(
                            (
                                a,
                                i_to_dir(i_sol),
                                Direction(k),
                                (trap, SignType.TRAP, Direction(j)),
                            )
                        )

    return inputs


def inputs_evaluation(
    path: Path,
    signs: dict[SignType, Signs],
    drawer: Callable,
    observations: np.ndarray,
    controller: BaseController,
    draw_inputs: bool = False,
    draw_individual_files: bool = False,
    draw_summary_file: bool = True,
    summary_file_ratio: float = 16 / 9,
):
    inputs = _all_inputs(signs)

    visuals, outputs = [], []
    df = pd.DataFrame(
        columns=[
            "Walls",
            "Type",
            "Sign",
            "SignType",
            "SignDirection",
            "Source",
            "Expected",
            "Action",
            "Success",
        ]
    )
    stats, totals = defaultdict(int), defaultdict(int)

    types = {3: "Dead-end", 2: "Corridor", 1: "Intersection"}

    NO_VALUE = "None"

    images = None
    with_visuals = len(observations.shape) > 1
    if with_visuals:
        all_signs = sum(signs.values(), [])
        images = {k: v for k, v in zip(all_signs, np_images(all_signs, observations.shape[0] - 2))}

    def _maybe_enum(_v):
        return _v.name if _v else NO_VALUE

    for walls, correct_action, prev_dir, oriented_sign in inputs:
        obs = observations.copy()
        obs.fill(0)

        visual = None
        if with_visuals and oriented_sign is not None:
            sign, _, direction = oriented_sign
            visual = images[sign][direction.value]

        drawer(obs, walls, visual, prev_dir)
        visuals.append(obs)

        action = controller(obs)
        outputs.append(tuple(action))
        # print(obs, oriented_sign)

        cell_type = types[walls.sum()]
        sign, sign_type, sign_direction = oriented_sign or (None, None, None)
        df.loc[len(df)] = [
            "".join(str(int(_w)) for _w in walls),
            cell_type,
            sign.to_string() if sign else NO_VALUE,
            _maybe_enum(sign_type),
            _maybe_enum(sign_direction),
            _maybe_enum(prev_dir),
            _maybe_enum(Maze.offsets_inv[tuple(correct_action)]),
            _maybe_enum(Maze.offsets_inv[tuple(action)]),
            action == correct_action,
        ]

        k = (cell_type, _maybe_enum(sign_type))
        stats[k] += action == correct_action
        totals[k] += 1

    df.to_csv(path.joinpath("inputs_recognition.csv"))

    stats_type, stats_signs = defaultdict(int), defaultdict(int)
    totals_type, totals_signs = defaultdict(int), defaultdict(int)
    for (kt, ks), v in stats.items():
        stats_type[kt] += v
        stats_signs[ks] += v
    for (kt, ks), v in totals.items():
        totals_type[kt] += v
        totals_signs[ks] += v
    stats_type = {k: stats_type[k] / totals_type[k] for k in stats_type}
    stats_signs = {k: stats_signs[k] / totals_signs[k] for k in stats_signs}

    total = len(df)
    stats = dict(
        counts=dict(stats),
        ratios={k: stats[k] / totals[k] for k in stats},
        ratios_per_sign=stats_signs,
        ratios_per_type=stats_type,
        totals=dict(totals),
        total=total,
        total_ratio=sum(stats.values()) / total,
    )

    if draw_inputs:
        __draw_inputs(
            path.joinpath("inputs.png"),
            inputs,
            visuals,
            outputs=None,
            individual_files=False,
            summary_file=True,
            summary_file_ratio=summary_file_ratio,
        )

    if draw_individual_files or draw_summary_file:
        __draw_inputs(
            path.joinpath("inputs_recognition.png"),
            inputs,
            visuals,
            outputs,
            individual_files=draw_individual_files,
            summary_file=draw_summary_file,
            summary_file_ratio=summary_file_ratio,
        )

    return df, stats
