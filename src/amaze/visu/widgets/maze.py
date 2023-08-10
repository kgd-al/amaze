import math
from pathlib import Path
from typing import Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QLineF
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QImage, QLinearGradient
from PyQt5.QtWidgets import QLabel

from amaze.simu.env.maze import Maze
from amaze.simu.robot import InputType
from amaze.simu.simulation import Simulation
from amaze.visu import resources
from amaze.visu.maze import MazePainter, Color, logger


class QtPainter(MazePainter):
    def __init__(self, p: QPainter):
        self.painter = p
        self.colors = {
            Color.WHITE: Qt.white,
            Color.BLACK: Qt.black,
            Color.RED: Qt.red,
            Color.GREEN: Qt.green,
            Color.BLUE: Qt.blue,
        }

    def draw_line(self, x0: float, y0: float, x1: float, y1: float, c: Color):
        self.painter.setPen(self.colors[c])
        self.painter.drawLine(QPointF(x0, y0),
                              QPointF(x1, y1))

    def fill_rect(self, x: float, y: float, w: float, h: float,
                  draw: Optional[Color], fill: Optional[Color],
                  alpha: float = 1):
        r = QRectF(x, y, w, h)
        if fill:
            fc = QColor(self.colors[fill])
            if alpha < 1:
                fc.setAlphaF(alpha)
            self.painter.fillRect(r, fc)
        if draw:
            self.painter.setPen(self.colors[draw])
            self.painter.drawRect(r)

    def fill_circle(self, x: float, y: float, r: float,
                    draw: Optional[Color], fill: Optional[Color]):
        path = QPainterPath()
        path.addEllipse(QRectF(x-r, y-r, 2 * r, 2 * r))
        self.painter.fillPath(path, self.colors[fill])

    def draw_image(self, x: float, y: float, w: float, h: float,
                   img: QImage):
        self.painter.drawImage(QRectF(x, y, w, h).toRect(), img, img.rect())

    def draw_flag(self, x: float, y: float, w: float, h: float, c: Color):
        path = QPainterPath()
        path.addRect(QRectF(x + .5 * w - 2, y + .2 * h, 3, .6 * h))
        path.moveTo(x + .5 * w + 1, y + .5 * h)
        path.lineTo(x + .8 * w, y + .65 * h)
        path.lineTo(x + .5 * w + 1, y + .8 * h)
        path.closeSubpath()

        self.painter.setPen(Qt.black)
        self.painter.fillPath(path, self.colors[c])
        self.painter.drawPath(path)


class MazeWidget(QLabel):
    def __init__(self,
                 simulation: Simulation,
                 config: Optional[dict[str, Any]] = None,
                 resolution: int = 15,
                 size: Tuple[int, int] = None):
        super().__init__("Generate a maze\n"
                         "using controls\n"
                         "on the right >>>")
        self._simulation = simulation
        self._scale = 12
        self._config = dict(
            solution=True,
            robot=True,
            dark=False
        )
        if config:
            self._config.update(config)

        self._vision = None
        self._clues, self._lures, self._traps = None, None, None
        self.set_resolution(resolution)

        if size:
            self.setFixedSize(*size)

        self._compute_scale()

    def set_simulation(self, simulation: Simulation):
        self._simulation = simulation
        self.set_resolution(simulation.data.vision)
        self.update()

    @staticmethod
    def __qt_images(size, maze):
        return (resources.qt_images(v, size) if v is not None else None
                for v in [maze.clues, maze.lures, maze.traps])

    def set_resolution(self, r: int):
        self._vision = r
        r -= 2
        if self._simulation.data.inputs is InputType.DISCRETE:
            m = self._simulation.maze
            r = min(self.width(), self.height()) // min(m.width, m.height)

        if maze := self._simulation.maze:
            self._clues, self._lures, self._traps = self.__qt_images(r, maze)

    def update_config(self, **kwargs):
        self._config.update(kwargs)
        self.update()

    @staticmethod
    def config_keys(): return ["solution", "robot", "dark"]

    def minimumSize(self) -> QSize:
        return 3 * QSize(self._simulation.maze.width,
                         self._simulation.maze.height)

    def resizeEvent(self, _):
        self._compute_scale()

    def showEvent(self, _):
        self._compute_scale()

    def update(self):
        self._compute_scale()
        super().update()

    def _compute_scale(self):
        maze = self._simulation.maze
        self._scale = round(min(self.width() / maze.width,
                                self.height() / maze.height))

    def paintEvent(self, e):
        maze = self._simulation.maze
        if maze is None:
            super().paintEvent(e)
            return

        painter = QPainter(self)

        sw = maze.width * self._scale + 2
        sh = maze.height * self._scale + 2
        painter.translate(.5 * (self.width() - sw), .5 * (self.height() - sh))

        painter.fillRect(QRectF(0, 0, sw, sh), self._background_color())

        self._render(painter)

        painter.end()

    def draw_to(self, path):
        img, painter = self._image_drawer()
        self._render(painter)
        self._save_image(path, img, painter)

    @classmethod
    def __render(cls, painter: QPainter, maze: Maze,
                 height: float, config: dict):
        painter.translate(1, height)
        painter.scale(1, -1)

        QtPainter(painter).render(maze, config)

    def _render(self, painter: QPainter):
        self.__render(
            painter, self._simulation.maze,
            self._simulation.maze.height * self._scale,
            dict(scale=self._scale,
                 clues=self._clues, lures=self._lures, traps=self._traps,
                 outputs=self._simulation.data.outputs,
                 solution=bool(self._config["solution"]),
                 robot=self._simulation.robot_dict()
                 if self._config["robot"] else None,
                 dark=self._config["dark"]))

    @staticmethod
    def __foreground_color(dark: bool): return Qt.white if dark else Qt.black

    def _foreground_color(self):
        return self.__foreground_color(self._config.get("dark", False))

    @staticmethod
    def __background_color(dark: bool): return Qt.black if dark else Qt.white

    def _background_color(self):
        return self.__background_color(self._config.get("dark", False))

    @staticmethod
    def __static_image_drawer(width: int, height: int,
                              fill: QColor,
                              img_format: QImage.Format = QImage.Format_RGB32)\
            -> Tuple[QImage, QPainter]:

        img = QImage(width, height, img_format)
        img.fill(fill)

        painter = QPainter(img)
        return img, painter

    def _image_drawer(self,
                      fill: Optional[QColor] = None,
                      img_format: QImage.Format = QImage.Format_RGB32) \
            -> Tuple[QImage, QPainter]:
        if fill is None:
            fill = self._background_color()
        return self.__static_image_drawer(
            self._scale * self._simulation.maze.width + 2,
            self._scale * self._simulation.maze.height + 2,
            fill, img_format)

    @staticmethod
    def _save_image(path: Union[str, Path],
                    img: QImage, painter: Optional[QPainter] = None):
        if painter:
            painter.end()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if img.save(str(path)):
            logger.debug(f"Wrote to {path}")
        else:
            logger.error(f"Error writing to {path}")

    @classmethod
    def plot_trajectory(cls, simulation: Simulation,
                        size: int,
                        trajectory: pd.DataFrame,
                        config: dict,
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
        width = size
        height = width * maze.height // maze.width
        scale = width / maze.width

        duplicates = (len(shortened_trajectory) < len(trajectory))
        cb_r = .2 if duplicates else 0
        cb_width = int(cb_r * width)
        # cb_height = int(cb_r * self.height())
        cb_margin = .1 * cb_width

        dark = config.get("dark", False)
        fill = cls.__background_color(dark)
        img, painter = cls.__static_image_drawer(
            width, height, fill, img_format=img_format)

        if duplicates:
            # Reserve 1/5 of the width
            painter.save()
            painter.scale(1-cb_r, 1-cb_r)
            painter.translate(0,
                              .5 * (height / (1-cb_r) - height))

        clues, lures, traps = cls.__qt_images(maze=maze, size=32)
        cls.__render(painter=painter, maze=maze, height=height,
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
                def color(_): return Qt.red
            else:
                def colormap(v): return QColor.fromRgbF(v, 1-v, 0)
                def color(n_): return colormap((n_ - min_count) / dif_count)
        else:
            def color(_): return cls.__foreground_color(dark)

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
            painter.setPen(cls.__foreground_color(dark))

            w, h = maze.width, maze.height
            m = cb_margin
            cb_h = s * h - 2 * cb_margin
            cb_w = .25 * cb_width
            painter.translate((1 - cb_r) * s * w, 0)
            gradient = QLinearGradient(0, 0, 0, 1)
            gradient.setColorAt(0, Qt.green)
            gradient.setColorAt(1, Qt.red)
            gradient.setCoordinateMode(QLinearGradient.ObjectMode)
            cb_box = QRectF(m, m, cb_w, cb_h)
            painter.fillRect(cb_box, gradient)
            painter.drawRect(cb_box)

            cb_ticks = min(5, max(dif_count, 2))
            tick_x, tick_x_w = cb_w + m, .1 * cb_width
            text_offset = .05 * cb_width
            text_rect = QRectF(tick_x + tick_x_w + text_offset, 0,
                               cb_width - tick_x - tick_x_w - text_offset - m,
                               img.height() / 5)

            fm = painter.fontMetrics()
            text_data = []
            for i_ in range(cb_ticks):
                u = i_ / (cb_ticks - 1)
                cb_y = dif_count * u + min_count
                cb_y = round(cb_y)
                text = str(cb_y)
                sx = text_rect.width() / fm.width(text)
                sy = img.height() / (5 * fm.height())
                ts = min(sx, sy)
                text_data.append([u, cb_y, text, ts])

            text_scale = min(sl[-1] for sl in text_data)
            for u, cb_y, text, _ in text_data:
                x_, y_ = tick_x, u * cb_h + m
                painter.drawLine(QLineF(x_, y_, x_ + tick_x_w, y_))

                text_rect.moveCenter(QPointF(text_rect.center().x(), y_))

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
            cls._save_image(path, img)
        else:
            return img
