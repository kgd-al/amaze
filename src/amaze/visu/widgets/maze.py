from pathlib import Path
from typing import Optional, Tuple, Union, Any

import pandas as pd
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QImage
from PyQt5.QtWidgets import QLabel

from amaze.simu.maze import Maze
from amaze.simu.simulation import Simulation
from amaze.simu.types import InputType, OutputType
from amaze.visu import resources
from amaze.visu.maze import MazePainter, Color, logger
from amaze.visu.resources import SignType
from amaze.visu.widgets import _trajectory_plotter, has_qt_application


class QtPainter(MazePainter):
    def __init__(self, p: QPainter, config: dict):
        self.painter = p
        dark = config['dark']
        if config['colorblind']:
            c1 = QColor("#de8f05")
            c2 = QColor("#0173b2")
            c3 = QColor("#029e73")
        else:
            c1 = Qt.red
            c2 = Qt.green
            c3 = Qt.blue
        self.colors = {
            Color.FOREGROUND: Qt.white if dark else Qt.black,
            Color.BACKGROUND: Qt.black if dark else Qt.white,
            Color.START: c1, Color.FINISH: c2, Color.PATH: c3,
            Color.COLOR1: c1, Color.COLOR2: c2, Color.COLOR3: c3
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

    def draw_start(self, x: float, y: float, w: float, h: float, c: Color):
        pass

    def draw_finish(self, x: float, y: float, w: float, h: float, c: Color):
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
                 width: Optional[int] = None):

        assert has_qt_application()
        super().__init__("Generate a maze\n"
                         "using controls\n"
                         "on the right >>>")
        self._simulation = simulation
        self._scale = 12
        self._config = self.default_config()
        if config:
            self._config.update(config)

        self._vision = None
        self._clues, self._lures, self._traps = None, None, None
        self.set_resolution(resolution)

        if width:
            self.setFixedSize(*self.__compute_size(simulation.maze, width))

        self._compute_scale()

    def set_simulation(self, simulation: Simulation):
        self._simulation = simulation
        self.set_resolution(simulation.data.vision)
        self.update()

    @staticmethod
    def __qt_images(size, maze):
        return (resources.qt_images(v, size)
                if (v := maze.signs[t]) is not None else None
                for t in SignType)

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
    def default_config():
        """ Returns the default rendering configuration.

        - solution: show the path the target (True)
        - robot: show the robot (True)
        - dark: use a dark background to see as the robot does (True)
        - colorblind: use a colorblind-friendly palette (False)
        - cycles: detect and simplify cycles when plotting trajectories (True)
        """
        return dict(
            solution=True,
            robot=True,
            dark=True,
            colorblind=False,
            cycles=True
        )

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

    @staticmethod
    def __compute_size(maze, width, square: bool = False):
        if square:
            return width, width
        else:
            return width, width * maze.height // maze.width

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

    def render_to(self, path):
        """ Render this widget to a file """
        img, painter = self._image_drawer()
        self._render(painter)
        return self._save_image(path, img, painter)

    def pretty_render(self, width: Optional[int] = None):
        img, painter = self._image_drawer(width=width)
        self._render(painter, width=width-2)
        painter.end()
        return img

    @classmethod
    def draw_to(cls, maze, path, size=None, **kwargs):
        """ Render the provided maze to a file (in png format).

        Additional arguments refer to the rendering configuration, see
        :meth:`default_config`.
        """
        width, height = cls.__compute_size(maze, size or maze.width * 15)
        scale = width / maze.width
        config = cls.default_config()
        config.update(dict(
            scale=scale,
            outputs=OutputType.DISCRETE,
            solution=True, robot=None, dark=True))
        config.update(kwargs)
        fill = cls.__background_color(config["dark"])
        img, painter = cls.__static_image_drawer(width + 2, height + 2, fill)
        clues, lures, traps = cls.__qt_images(maze=maze, size=32)
        config.update(dict(clues=clues, lures=lures, traps=traps))
        cls.__render(painter=painter, maze=maze, height=height, config=config)
        return cls._save_image(path, img, painter)

    @classmethod
    def __render(cls, painter: QPainter, maze: Maze,
                 height: float, config: dict):
        painter.translate(1, height)
        painter.scale(1, -1)

        QtPainter(painter, config).render(maze, config)

    def _render(self, painter: QPainter, width: Optional[int] = None):
        config = dict(self._config)
        config.update(dict(
            scale=self._scale,
            clues=self._clues, lures=self._lures, traps=self._traps,
            outputs=self._simulation.data.outputs,
            solution=bool(self._config["solution"]),
            robot=self._simulation.robot_dict()
            if self._config["robot"] else None))

        if width is not None:
            config['scale'] = width / self._simulation.maze.width

        self.__render(
            painter, self._simulation.maze,
            self._simulation.maze.height * config['scale'],
            config)

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

        assert has_qt_application()
        img = QImage(width, height, img_format)
        img.fill(fill)

        painter = QPainter(img)
        return img, painter

    def _image_drawer(self,
                      fill: Optional[QColor] = None,
                      width: Optional[int] = None,
                      img_format: QImage.Format = QImage.Format_RGB32) \
            -> Tuple[QImage, QPainter]:
        if fill is None:
            fill = self._background_color()

        if width is None:
            width = self._scale * self._simulation.maze.width + 2
            height = self._scale * self._simulation.maze.height + 2
        else:
            scale = (width-2) / self._simulation.maze.width
            height = round(self._simulation.maze.height * scale) + 2

        return self.__static_image_drawer(
            width, height,
            fill, img_format)

    @staticmethod
    def _save_image(path: Union[str, Path],
                    img: QImage, painter: Optional[QPainter] = None):
        if painter:
            painter.end()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = img.save(str(path))
        if ok:
            logger.debug(f"Wrote to {path}")
        else:
            logger.error(f"Error writing to {path}")
        return ok

    @classmethod
    def plot_trajectory(cls, simulation: Simulation,
                        size: int,
                        trajectory: Optional[pd.DataFrame] = None,
                        config: Optional[dict] = None,
                        path: Optional[Path | str] = None,
                        verbose: int = 0,
                        side: int = 0,
                        square: bool = False,
                        img_format: QImage.Format = QImage.Format_RGB32) \
            -> Optional[QImage]:
        """ Plots a trajectory stored in the provided simulation to the
        requested file.

        :param simulation: the simulation to plot from
        :param size: the width of the generated image (height is deduced)
        :param trajectory: the trajectory (in dataframe format)
        :param path: where to save the trajectory to (or None to get the image)
        :param verbose: whether to provide additional information
        :param side: where to put the color bar (<0: left, >0: right, 0: auto)
        :param square: whether to generate a square image of size max(w,h)^2
        :param img_format: QImage.Format to use for the underlying QImage
        :param config: kw configuration values (see default_config())
        """

        funcs = dict(
            background=cls.__background_color,
            foreground=cls.__foreground_color,
            size=cls.__compute_size,
            image_painter=cls.__static_image_drawer,
            qt_images=cls.__qt_images,
            render=cls.__render,
            save=cls._save_image
        )

        if trajectory is None:
            trajectory = simulation.trajectory
        assert len(trajectory) > 0

        _config = cls.default_config()
        _config["robot"] = False
        if config is not None:
            _config.update(config)

        return _trajectory_plotter.plot_trajectory(
            simulation=simulation,
            size=size,
            trajectory=trajectory,
            config=_config,
            functions=funcs,
            verbose=verbose,
            side=side,
            square=square,
            path=path,
            img_format=img_format)
