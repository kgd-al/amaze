from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF
from PyQt5.QtGui import (
    QImage,
    QColor,
    QPainterPath,
    QLinearGradient,
    QFontMetrics,
)

from ...simu.simulation import Simulation

# Not as informative
# def _plot_trajectory_duplicates(
#     simulation: Simulation,
#     size: int,
#     trajectory: pd.DataFrame,
#     config: dict,
#     functions: dict,
#     path: Optional[Path] = None,
#     img_format: QImage.Format = QImage.Format_RGB32,
# ) -> Optional[QImage]:
#     """
#     Plots the agents trajectory while showing the number of repetitions as colormap
#
#     The dataframe's columns must be [px, py, ax, ay, r], with:
#         - px, py the robot's position
#         - ax, ay the action
#         - r the resulting reward
#     :param simulation: the simulation to plot from
#     :param size: the width of the generated image (height is deduced)
#     :param trajectory: the trajectory (in dataframe format)
#     :param path: where to save the trajectory to (or None to get the image)
#     :param img_format: QImage.Format to use for the underlying QImage
#     :param config: kw configuration values (see config_keys())
#     """
#
#     _trajectory = []
#
#     shortened_trajectory = trajectory.groupby(
#         trajectory.columns.tolist()[:-1], as_index=False
#     ).size()
#
#     maze = simulation.maze
#     width, height = functions["size"](maze, size)
#     scale = width / maze.width
#
#     duplicates = len(shortened_trajectory) < len(trajectory)
#     cb_r = 0.2 if duplicates else 0
#     cb_width = int(cb_r * width)
#     # cb_height = int(cb_r * self.height())
#     cb_margin = 0.1 * cb_width
#
#     dark = config.get("dark", False)
#     fill = functions["background"](dark)
#     img, painter = functions["image_painter"](width + 2, height + 2, fill,
#                                               img_format=img_format)
#
#     if duplicates:
#         # Reserve 1/5 of the width
#         painter.save()
#         painter.scale(1 - cb_r, 1 - cb_r)
#         painter.translate(0, 0.5 * (height / (1 - cb_r) - height))
#
#     clues, lures, traps = functions["qt_images"](maze=maze, size=32)
#     functions["render"](
#         painter=painter,
#         maze=maze,
#         height=height,
#         config=dict(
#             scale=scale,
#             clues=clues,
#             lures=lures,
#             traps=traps,
#             outputs=simulation.data.outputs,
#             solution=config["solution"],
#             robot=simulation.robot_dict() if config["robot"] else None,
#             dark=config["dark"],
#         ),
#     )
#
#     min_count, max_count = np.quantile(shortened_trajectory.iloc[:, -1], [0, 1])
#     dif_count = max_count - min_count
#
#     rotations = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}
#
#     if duplicates:
#         if dif_count == 0:
#
#             def color(_):
#                 return Qt.red
#
#         else:
#
#             def colormap(v):
#                 return QColor.fromRgbF(v, 1 - v, 0)
#
#             def color(n_):
#                 return colormap((n_ - min_count) / dif_count)
#
#     else:
#
#         def color(_):
#             return functions["foreground"](dark)
#
#     for x, y, i, j, n in shortened_trajectory.itertuples(index=False):
#         _trajectory.append(((x, y), rotations[int(i), int(j)], color(n)))
#
#     s = scale
#     arrow = QPainterPath()
#     arrow.moveTo(0, -0.1 * s)
#     arrow.lineTo(0, 0.1 * s)
#     arrow.lineTo(0.3 * s, 0.1 * s)
#     arrow.lineTo(0.3 * s, 0.2 * s)
#     arrow.lineTo(0.5 * s, 0.0 * s)
#     arrow.lineTo(0.3 * s, -0.2 * s)
#     arrow.lineTo(0.3 * s, -0.1 * s)
#     arrow.lineTo(0, -0.1 * s)
#     arrow.closeSubpath()
#
#     for (x_, y_), a, c in _trajectory:
#         painter.save()
#         painter.translate(x_ * s, y_ * s)
#         painter.rotate(a)
#         painter.fillPath(arrow, c)
#         painter.setPen(QColor(c).darker())
#         painter.drawPath(arrow)
#         painter.restore()
#
#     if duplicates:
#         painter.restore()  # to before the space saved for the color bar
#         painter.setPen(functions["foreground"](dark))
#
#         w, h = maze.width, maze.height
#         m = cb_margin
#         cb_h = s * h - 2 * cb_margin
#         cb_w = 0.25 * cb_width
#         painter.translate((1 - cb_r) * s * w, 0)
#         gradient = QLinearGradient(0, 0, 0, 1)
#         gradient.setColorAt(0, Qt.red)
#         gradient.setColorAt(1, Qt.green)
#         gradient.setCoordinateMode(QLinearGradient.ObjectMode)
#         cb_box = QRectF(m, m, cb_w, cb_h)
#         painter.fillRect(cb_box, gradient)
#         painter.drawRect(cb_box)
#
#         fm = painter.fontMetrics()
#         cb_ticks = min(5, max(dif_count, 2))
#         tick_x, tick_x_w = cb_w + m, 0.1 * cb_width
#         text_offset = 0.05 * cb_width
#         text_rect = QRectF(
#             tick_x + tick_x_w + text_offset,
#             0,
#             cb_width - tick_x - tick_x_w - text_offset - m,
#             fm.height(),
#         )
#
#         text_data = []
#         for i_ in range(cb_ticks):
#             u = i_ / (cb_ticks - 1)
#             cb_y = dif_count * u + min_count
#             cb_y = round(cb_y)
#             text = str(cb_y)
#             sx = text_rect.width() / fm.width(text)
#             ts = min(sx, fm.height())
#             text_data.append([u, cb_y, text, ts])
#
#         text_scale = min(*(sl[-1] for sl in text_data), 1)
#         for u, _cb_y, text, _ in text_data:
#             x_, y_ = tick_x, (1 - u) * cb_h + m
#             painter.drawLine(QLineF(x_, y_, x_ + tick_x_w, y_))
#
#             text_rect.moveCenter(QPointF(text_rect.center().x(), y_))
#             if text_rect.top() < 0:
#                 text_rect.moveTop(0)
#             if (height - text_rect.bottom()) < 0:
#                 text_rect.moveBottom(height)
#
#             painter.save()
#             painter.translate(text_rect.center())
#             painter.scale(text_scale, text_scale)
#             painter.translate(-text_rect.center())
#             painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, text)
#             painter.restore()
#             # painter.drawRect(text_rect)
#
#     painter.end()
#     if path:
#         functions["save"](path, img)
#     else:
#         return img


def _plot_trajectory_value(
    simulation: Simulation,
    size: int,
    trajectory: pd.DataFrame,
    config: dict,
    functions: dict,
    path: Optional[Path] = None,
    verbose: int = 0,
    side: int = 0,
    square: bool = False,
    force_overlay: bool = False,
    img_format: QImage.Format = QImage.Format_RGB32,
) -> Optional[QImage]:

    # verbose = 0
    if verbose > 0:
        if verbose > 5:  # pragma: no cover
            print()
            print("=" * 80)
        print(f"Verbose debugging of {__name__}._plot_trajectory_value")
    # side = -1

    maze = simulation.maze
    rewards = simulation.rewards

    if side == 0:
        side = -1 if maze.start[0] == 0 else 1

    c_reward = 0
    solution_index = 0
    backtrack_stack = []
    state_action = dict()

    r_min = simulation.minimal_reward
    r_range = simulation.optimal_reward - r_min
    def v_norm(_v): return round((v - r_min) / r_range, 3)

    for i, px, py, ax, ay, r in trajectory.itertuples():
        if verbose > 2:  # pragma: no cover
            print("===")

        # Compute value
        c_reward += r
        required_steps = 0

        cell = (int(px), int(py))
        if i < len(trajectory) - 1:
            next_cell = tuple(int(_p) for _p in trajectory[["px", "py"]].iloc[i+1])
            if verbose > 2:  # pragma: no cover
                print(f"{cell=}, {next_cell=}")
            if cell == next_cell:
                if verbose > 2:
                    print("Hitting a wall")
                pass
            elif next_cell == maze.solution[solution_index + 1]:
                if verbose > 2:  # pragma: no cover
                    print("Still on track")
                solution_index += 1
            elif next_cell == maze.solution[solution_index]:
                if verbose > 2:  # pragma: no cover
                    print("Backward")
                solution_index -= 1
            else:
                if verbose > 2:  # pragma: no cover
                    print("Off track")
                required_steps += len(backtrack_stack)
                if backtrack_stack and next_cell == backtrack_stack[-1]:
                    backtrack_stack.pop()
                # elif cell != maze.solution[solution_index]:
                else:
                    backtrack_stack.append(cell)
            if verbose > 2:  # pragma: no cover
                print(len(backtrack_stack), backtrack_stack)

        elif simulation.success():
            solution_index += 1

        required_steps += len(maze.solution) - solution_index

        v = c_reward
        possible_steps = simulation.deadline - i
        steps = min(required_steps, possible_steps)
        can_reach_goal = (r < 0 and required_steps < possible_steps)
        if can_reach_goal:
            v += rewards.finish

        v += (steps - 1) * rewards.timestep
        v = v_norm(v)
        assert 0 <= v <= 1, v

        k = ((px, py), (ax, ay))
        state_action[k] = v

    values = set(state_action.values())
    trivial_trajectory = len(values) == 1

    needs_overlay = ((values != {1}) and verbose > 0) or force_overlay
    if verbose > 1:
        print(f"{trivial_trajectory=} {needs_overlay=}")
    # pprint.pprint(values)

    width, height = functions["size"](maze, size, square)
    scale = min(width / maze.width, height / maze.height)

    cb_r = 0.2 if needs_overlay else 0
    cb_width = int(cb_r * width)
    # cb_height = int(cb_r * self.height())
    cb_margin = 0.1 * cb_width

    fill = config["background"]
    img, painter = functions["image_painter"](
        width + 2, height + 2, fill, img_format=img_format
    )

    x_offset = 0
    if needs_overlay:
        # Reserve 1/5 of the width
        painter.save()
        painter.scale(1 - cb_r, 1 - cb_r)
        x_offset = 0 if side > 0 else cb_width / (1 - cb_r)
        y_offset = 0.5 * (height / (1 - cb_r) - height)
        painter.translate(x_offset, y_offset)

    force_square = square and maze.width != maze.height
    if force_square:
        painter.save()
        m_x_offset, m_y_offset = 0, 0
        if maze.start[0] == 0 and (dx := maze.height - maze.width) > 0:
            m_x_offset = dx
        if maze.start[1] == 0 and (dy := maze.width - maze.height) > 0:
            m_y_offset = dy
        painter.translate(m_x_offset * scale, -m_y_offset * scale)

    clues, lures, traps = functions["qt_images"](maze=maze, size=32)
    render_config = config.copy()
    render_config.update(
        dict(
            scale=scale,
            clues=clues,
            lures=lures,
            traps=traps,
            outputs=simulation.data.outputs,
            robot=simulation.robot.to_dict() if config["robot"] else None,
        )
    )
    functions["render"](painter=painter, maze=maze, height=height, config=render_config)

    rotations = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}

    if needs_overlay:
        def colormap(v_): return QColor.fromHsvF(v_ / 6, 1, 1)
        def color(v_): return Qt.green if v_ == 1 else colormap(v_)

    else:
        def color(_): return Qt.green

    _trajectory = []
    for (((x, y), (i, j)), v) in state_action.items():
        _trajectory.append(((x, y), rotations[int(i), int(j)], color(v)))

    s = scale
    arrow = QPainterPath()
    arrow.moveTo(-0.05 * s, -0.1 * s)
    arrow.moveTo(0.0, 0.0)
    arrow.lineTo(-0.05 * s, 0.1 * s)
    arrow.lineTo(0.3 * s, 0.1 * s)
    arrow.lineTo(0.3 * s, 0.2 * s)
    arrow.lineTo(0.5 * s, 0.0 * s)
    arrow.lineTo(0.3 * s, -0.2 * s)
    arrow.lineTo(0.3 * s, -0.1 * s)
    arrow.lineTo(-0.05 * s, -0.1 * s)
    arrow.closeSubpath()

    for (x_, y_), a, c in _trajectory:
        painter.save()
        painter.translate(x_ * s, y_ * s)
        painter.rotate(a)
        painter.fillPath(arrow, c)
        painter.setPen(QColor(c).darker())
        painter.drawPath(arrow)
        painter.restore()

    if needs_overlay:
        if not simulation.success():
            (x_, y_), a, c = _trajectory[-1]
            painter.save()
            painter.translate(x_ * s, y_ * s)
            painter.rotate(a)
            painter.fillRect(QRectF(0.4 * s, -0.5 * s + 1, 0.1 * s, s - 2), Qt.red)
            painter.restore()

        if force_square:
            painter.restore()

        painter.restore()  # to before the space saved for the color bar
        painter.setPen(config["foreground"])

        if verbose:
            painter.drawText(
                QRectF(x_offset, 0, width - cb_width, 0.5 * cb_width),
                Qt.AlignCenter,
                "Expected return",
            )
            actions, unique_actions = len(trajectory), len(state_action)
            if actions != unique_actions:
                cycle_legend = f"{unique_actions} unique actions out of {actions}"\
                               f" ({100 * unique_actions / actions:.2g}%)"
            else:
                cycle_legend = f"{actions} actions"

            painter.drawText(
                QRectF(
                    x_offset,
                    height - 0.5 * cb_width,
                    width - cb_width,
                    0.5 * cb_width,
                ),
                Qt.AlignCenter, cycle_legend
            )

    elif force_square:
        painter.restore()

    if needs_overlay:
        # w, h = maze.width, maze.height
        w, h = width, height
        m = cb_margin
        # cb_h = s * h - 2 * cb_margin
        cb_h = h - 2 * cb_margin
        cb_w = 0.25 * cb_width

        if side == 1:
            painter.translate((1 - cb_r) * w, 0)
        else:
            painter.translate(cb_width, 0)
            painter.scale(-1, 1)

        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setColorAt(0, Qt.green)
        gradient.setColorAt(.05, colormap(1))
        gradient.setColorAt(1, colormap(0))
        gradient.setCoordinateMode(QLinearGradient.ObjectMode)
        cb_box = QRectF(m, m, cb_w, cb_h)
        painter.fillRect(cb_box, gradient)
        painter.drawRect(cb_box)

        fm: QFontMetrics = painter.fontMetrics()
        cb_ticks = 5  # min(5, max(diff_v, 2))
        tick_x, tick_x_w = cb_w + m, 0.1 * cb_width
        text_offset = 0.05 * cb_width
        text_rect = QRectF(
            tick_x + tick_x_w + text_offset,
            0,
            cb_width - tick_x - tick_x_w - text_offset,
            0,
        )

        text_data = []
        for _i in range(cb_ticks):
            u = _i / (cb_ticks - 1)
            cb_y = u
            # cb_y = round(cb_y)
            text = f"{cb_y:.2g}"
            if verbose > 1:
                print(cb_y, text, fm.width(text))
            bb = fm.tightBoundingRect(text)
            text_rect.setHeight(max(text_rect.height(), bb.height()))
            sx = text_rect.width() / max(bb.width(), fm.width(text))
            ts = min(sx, fm.height())
            text_data.append([u, cb_y, text, ts])

        text_scale = min(*(sl[-1] for sl in text_data), 1)
        text_scaled_rect = QRectF(
            -0.5 * text_rect.width() / text_scale,
            -0.5 * text_rect.height() / text_scale,
            text_rect.width() / text_scale,
            text_rect.height() / text_scale,
        )
        text_align = (Qt.AlignRight if side < 0 else Qt.AlignLeft) | Qt.AlignVCenter
        for u, _cb_y, text, _ in text_data:
            x_, y_ = tick_x, (1 - u) * cb_h + m
            painter.drawLine(QLineF(x_, y_, x_ + tick_x_w, y_))

            text_rect.moveCenter(QPointF(text_rect.center().x(), y_))
            if text_rect.top() < 0:
                text_rect.moveTop(0)
            if (height - text_rect.bottom()) < 0:
                text_rect.moveBottom(height)

            painter.save()
            painter.translate(text_rect.center())
            painter.scale(side * text_scale, text_scale)
            painter.drawText(text_scaled_rect, text_align, text)
            # painter.drawRect(text_scaled_rect)
            painter.restore()
            # painter.drawRect(text_rect)

    painter.end()

    if path:
        functions["save"](path, img)
    else:  # pragma: no cover
        return img


plot_trajectory = _plot_trajectory_value
