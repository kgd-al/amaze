import pprint
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF
from PyQt5.QtGui import QImage, QColor, QPainterPath, QLinearGradient, QFontMetrics

from amaze.simu.simulation import Simulation


def _plot_trajectory_duplicates(
        simulation: Simulation,
        size: int,
        trajectory: pd.DataFrame,
        config: dict, functions: dict,
        path: Optional[Path] = None,
        img_format: QImage.Format = QImage.Format_RGB32) \
            -> Optional[QImage]:
    """
    The dataframe's columns must be [px, py, ax, ay, r], with:
        - px, py the robot's position
        - ax, ay the action
        - r the resulting reward
    :param simulation: the simulation to plot from
    :param size: the width of the generated image (height is deduced)
    :param trajectory: the trajectory (in dataframe format)
    :param path: where to save the trajectory to (or None to get the image)
    :param img_format: QImage.Format to use for the underlying QImage
    :param config: kw configuration values (see config_keys())
    """

    _trajectory = []

    shortened_trajectory = \
        trajectory.groupby(trajectory.columns.tolist()[:-1],
                           as_index=False).size()

    maze = simulation.maze
    width, height = functions['size'](maze, size)
    scale = width / maze.width

    duplicates = (len(shortened_trajectory) < len(trajectory))
    cb_r = .2 if duplicates else 0
    cb_width = int(cb_r * width)
    # cb_height = int(cb_r * self.height())
    cb_margin = .1 * cb_width

    dark = config.get("dark", False)
    fill = functions['background'](dark)
    img, painter = functions['image_painter'](
        width + 2, height + 2, fill, img_format=img_format)

    if duplicates:
        # Reserve 1/5 of the width
        painter.save()
        painter.scale(1 - cb_r, 1 - cb_r)
        painter.translate(0,
                          .5 * (height / (1 - cb_r) - height))

    clues, lures, traps = functions['qt_images'](maze=maze, size=32)
    functions['render'](painter=painter, maze=maze, height=height,
                        config=dict(
                             scale=scale,
                             clues=clues, lures=lures, traps=traps,
                             outputs=simulation.data.outputs,
                             solution=config["solution"],
                             robot=simulation.robot_dict()
                             if config["robot"] else None,
                             dark=config["dark"]))

    min_count, max_count = \
        np.quantile(shortened_trajectory.iloc[:, -1], [0, 1])
    dif_count = max_count - min_count

    rotations = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}

    if duplicates:
        if dif_count == 0:
            def color(_):
                return Qt.red
        else:
            def colormap(v):
                return QColor.fromRgbF(v, 1 - v, 0)

            def color(n_):
                return colormap((n_ - min_count) / dif_count)
    else:
        def color(_):
            return functions['foreground'](dark)

    for x, y, i, j, n in shortened_trajectory.itertuples(index=False):
        _trajectory.append(((x, y), rotations[int(i), int(j)],
                            color(n)))

    s = scale
    arrow = QPainterPath()
    arrow.moveTo(0, -.1 * s)
    arrow.lineTo(0, .1 * s)
    arrow.lineTo(.3 * s, .1 * s)
    arrow.lineTo(.3 * s, .2 * s)
    arrow.lineTo(.5 * s, .0 * s)
    arrow.lineTo(.3 * s, -.2 * s)
    arrow.lineTo(.3 * s, -.1 * s)
    arrow.lineTo(0, -.1 * s)
    arrow.closeSubpath()

    for (x_, y_), a, c in _trajectory:
        painter.save()
        painter.translate(x_ * s, y_ * s)
        painter.rotate(a)
        painter.fillPath(arrow, c)
        painter.setPen(QColor(c).darker())
        painter.drawPath(arrow)
        painter.restore()

    if duplicates:
        painter.restore()  # to before the space saved for the color bar
        painter.setPen(functions['foreground'](dark))

        w, h = maze.width, maze.height
        m = cb_margin
        cb_h = s * h - 2 * cb_margin
        cb_w = .25 * cb_width
        painter.translate((1 - cb_r) * s * w, 0)
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setColorAt(0, Qt.red)
        gradient.setColorAt(1, Qt.green)
        gradient.setCoordinateMode(QLinearGradient.ObjectMode)
        cb_box = QRectF(m, m, cb_w, cb_h)
        painter.fillRect(cb_box, gradient)
        painter.drawRect(cb_box)

        fm = painter.fontMetrics()
        cb_ticks = min(5, max(dif_count, 2))
        tick_x, tick_x_w = cb_w + m, .1 * cb_width
        text_offset = .05 * cb_width
        text_rect = QRectF(tick_x + tick_x_w + text_offset, 0,
                           cb_width - tick_x - tick_x_w - text_offset - m,
                           fm.height())

        text_data = []
        for i_ in range(cb_ticks):
            u = i_ / (cb_ticks - 1)
            cb_y = dif_count * u + min_count
            cb_y = round(cb_y)
            text = str(cb_y)
            sx = text_rect.width() / fm.width(text)
            ts = min(sx, fm.height())
            text_data.append([u, cb_y, text, ts])

        text_scale = min(*(sl[-1] for sl in text_data), 1)
        for u, cb_y, text, _ in text_data:
            x_, y_ = tick_x, (1 - u) * cb_h + m
            painter.drawLine(QLineF(x_, y_, x_ + tick_x_w, y_))

            text_rect.moveCenter(QPointF(text_rect.center().x(), y_))
            if text_rect.top() < 0:
                text_rect.moveTop(0)
            if (height - text_rect.bottom()) < 0:
                text_rect.moveBottom(height)

            painter.save()
            painter.translate(text_rect.center())
            painter.scale(text_scale, text_scale)
            painter.translate(-text_rect.center())
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter,
                             text)
            painter.restore()
            # painter.drawRect(text_rect)

    painter.end()
    if path:
        functions['save'](path, img)
    else:
        return img


def _plot_trajectory_value(
        simulation: Simulation,
        size: int,
        trajectory: pd.DataFrame,
        config: dict, functions: dict,
        path: Optional[Path] = None,
        verbose=True,
        img_format: QImage.Format = QImage.Format_RGB32) \
            -> Optional[QImage]:

    maze = simulation.maze
    rewards = simulation.rewards


    duplicates = len(trajectory.groupby(trajectory.columns.tolist()[:-1])) < len(trajectory)
    print("One-liner cycle detection:", duplicates)

    # First look for cycles
    data = pd.DataFrame(columns=["i", "j", "cr", "steps", "value"])
    c_reward = 0
    values = []
    solution_index = 0
    backtrack_stack = []
    visited = set()
    state_action = set()
    cycle = None
    for i, px, py, ax, ay, r in trajectory.itertuples():
        print("===")
        # Detect cycles
        k = ((px, py), (ax, ay))
        if k in state_action:
            cycle = i
            print("Cycle detected")
            break
        state_action.add(k)

        # Compute value
        c_reward += r
        v = c_reward + rewards.finish

        steps = 0

        cell = (int(px), int(py))
        next_cell = (int(px+ax), int(py+ay))
        print(f"{cell=}, {next_cell=}")
        if next_cell == maze.solution[solution_index+1]:
            print("Still on track")
            solution_index += 1
        elif next_cell == maze.solution[solution_index]:
            print("Backward")
            solution_index -= 1
        else:
            print("Off track")
            steps += len(backtrack_stack)
            if backtrack_stack and next_cell == backtrack_stack[-1]:
                backtrack_stack.pop()
            # elif cell != maze.solution[solution_index]:
            else:
                backtrack_stack.append(cell)
        print(len(backtrack_stack), backtrack_stack)
        steps += (len(maze.solution) - solution_index - 1)

        v += steps * rewards.timestep
        v /= len(maze.solution)
        data.loc[len(data)] = [*cell, c_reward, steps, v]
        values.append(v)
    print(trajectory.merge(data, left_index=True, right_index=True))

    print(f"Cycle: [0:{cycle}]")
    # pprint.pprint(values)

    width, height = functions['size'](maze, size)
    scale = width / maze.width

    cb_r = .2 if cycle else 0
    cb_width = int(cb_r * width)
    # cb_height = int(cb_r * self.height())
    cb_margin = .1 * cb_width

    dark = config.get("dark", False)
    fill = functions['background'](dark)
    img, painter = functions['image_painter'](
        width + 2, height + 2, fill, img_format=img_format)

    if cycle:
        # Reserve 1/5 of the width
        painter.save()
        painter.scale(1 - cb_r, 1 - cb_r)
        painter.translate(0,
                          .5 * (height / (1 - cb_r) - height))

    clues, lures, traps = functions['qt_images'](maze=maze, size=32)
    functions['render'](painter=painter, maze=maze, height=height,
                        config=dict(
                             scale=scale,
                             clues=clues, lures=lures, traps=traps,
                             outputs=simulation.data.outputs,
                             solution=config["solution"],
                             robot=simulation.robot_dict()
                             if config["robot"] else None,
                             dark=config["dark"]))

    min_v, max_v = np.quantile(values, [0, 1])
    diff_v = max_v - min_v
    assert diff_v > 0
    print("Value range:", min_v, max_v, diff_v)
    #
    rotations = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}

    if cycle:
        def colormap(v_): return QColor.fromRgbF(1 - v_, .5 * v_, 0)

        def color(v_):
            return Qt.green if v_ == max_v \
                else colormap((v_ - min_v) / diff_v)
    else:
        def color(_): return Qt.green

    assert not cycle or len(values) == cycle, f"{len(values)} != {cycle}"
    assert cycle or len(values) == len(trajectory),\
        f"{len(values)} != {len(trajectory)}"

    _trajectory = []
    for (x, y, i, j, _), cr in (
            zip(trajectory.itertuples(index=False), values)):
        _trajectory.append(((x, y), rotations[int(i), int(j)],
                            color(cr)))

    s = scale
    arrow = QPainterPath()
    arrow.moveTo(-.05*s, -.1 * s)
    arrow.moveTo(0.0, 0.0)
    arrow.lineTo(-.05*s, .1 * s)
    arrow.lineTo(.3 * s, .1 * s)
    arrow.lineTo(.3 * s, .2 * s)
    arrow.lineTo(.5 * s, .0 * s)
    arrow.lineTo(.3 * s, -.2 * s)
    arrow.lineTo(.3 * s, -.1 * s)
    arrow.lineTo(-.05*s, -.1 * s)
    arrow.closeSubpath()

    for (x_, y_), a, c in _trajectory:
        painter.save()
        painter.translate(x_ * s, y_ * s)
        painter.rotate(a)
        painter.fillPath(arrow, c)
        painter.setPen(QColor(c).darker())
        painter.drawPath(arrow)
        painter.restore()

    if cycle:
        (x_, y_), a, c = _trajectory[-1]
        painter.save()
        painter.translate(x_ * s, y_ * s)
        painter.rotate(a)
        painter.fillRect(QRectF(.4 * s, -.5 * s + 1, .1*s, s - 2), Qt.red)
        painter.restore()

        painter.restore()  # to before the space saved for the color bar
        painter.setPen(functions['foreground'](dark))

        if verbose:
            painter.drawText(QRectF(0, 0, width - cb_width, .5*cb_width),
                             Qt.AlignCenter, "Expected return")
            painter.drawText(QRectF(0, height-.5*cb_width,
                                    width - cb_width, .5*cb_width),
                             Qt.AlignCenter,
                             f"First cycle ({100*cycle/len(trajectory):.2g}%)")

        w, h = maze.width, maze.height
        m = cb_margin
        cb_h = s * h - 2 * cb_margin
        cb_w = .25 * cb_width
        painter.translate((1 - cb_r) * s * w, 0)
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setColorAt(0, Qt.green)
        gradient.setColorAt((max_v - sorted(set(values))[-2]) / diff_v,
                            Qt.darkGreen)
        gradient.setColorAt(1, Qt.red)
        gradient.setCoordinateMode(QLinearGradient.ObjectMode)
        cb_box = QRectF(m, m, cb_w, cb_h)
        painter.fillRect(cb_box, gradient)
        painter.drawRect(cb_box)

        fm: QFontMetrics = painter.fontMetrics()
        cb_ticks = 5#min(5, max(diff_v, 2))
        tick_x, tick_x_w = cb_w + m, .1 * cb_width
        text_offset = .05 * cb_width
        text_rect = QRectF(tick_x + tick_x_w + text_offset, 0,
                           cb_width - tick_x - tick_x_w - text_offset,
                           0)

        text_data = []
        for i_ in range(cb_ticks):
            u = i_ / (cb_ticks - 1)
            cb_y = diff_v * u + min_v
            # cb_y = round(cb_y)
            text = f"{cb_y:.2g}"
            print(cb_y, text, fm.width(text))
            bb = fm.tightBoundingRect(text)
            text_rect.setHeight(max(text_rect.height(), bb.height()))
            sx = text_rect.width() / max(bb.width(), fm.width(text))
            ts = min(sx, fm.height())
            text_data.append([u, cb_y, text, ts])

        text_scale = min(*(sl[-1] for sl in text_data), 1)
        text_scaled_rect = QRectF(
            -.5 * text_rect.width() / text_scale,
            -.5 * text_rect.height() / text_scale,
            text_rect.width() / text_scale,
            text_rect.height() / text_scale,
        )
        for u, cb_y, text, _ in text_data:
            x_, y_ = tick_x, (1 - u) * cb_h + m
            painter.drawLine(QLineF(x_, y_, x_ + tick_x_w, y_))

            text_rect.moveCenter(QPointF(text_rect.center().x(), y_))
            if text_rect.top() < 0:
                text_rect.moveTop(0)
            if (height - text_rect.bottom()) < 0:
                text_rect.moveBottom(height)

            painter.save()
            painter.translate(text_rect.center())
            painter.scale(text_scale, text_scale)
            painter.drawText(text_scaled_rect, Qt.AlignLeft | Qt.AlignVCenter,
                             text)
            # painter.drawRect(text_scaled_rect)
            painter.restore()
            # painter.drawRect(text_rect)

    painter.end()
    if path:
        functions['save'](path, img)
    else:
        return img


plot_trajectory = _plot_trajectory_value
