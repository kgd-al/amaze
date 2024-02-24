from abc import ABC, abstractmethod
from enum import Enum, auto
from logging import getLogger
from typing import Optional

from amaze.simu.maze import Maze
from amaze.simu.robot import Robot
from amaze.simu.types import OutputType
from amaze.visu.resources import SignType

logger = getLogger(__name__)


class Color(Enum):
    BACKGROUND = auto()
    FOREGROUND = auto()
    START = auto()
    FINISH = auto()
    PATH = auto()
    COLOR1 = auto()
    COLOR2 = auto()
    COLOR3 = auto()


class MazePainter(ABC):
    @abstractmethod
    def draw_line(self, x0: float, y0: float, x1: float, y1: float, c: Color):
        pass

    @abstractmethod
    def fill_rect(self, x: float, y: float, w: float, h: float,
                  draw: Color, fill: Color, alpha: float = 1):
        pass

    @abstractmethod
    def fill_circle(self, x: float, y: float, r: float,
                    draw: Optional[Color], fill: Optional[Color]):
        pass

    @abstractmethod
    def draw_image(self, x: float, y: float, w: float, h: float, img):
        pass

    @abstractmethod
    def draw_start(self, x: int, y: int, w: float, h: float, c: Color):
        pass

    @abstractmethod
    def draw_finish(self, x: int, y: int, w: float, h: float, c: Color):
        pass

    def render(self, maze: Maze, options: dict):
        # noinspection PyPep8Naming
        EAST, NORTH, WEST, SOUTH = [d for d in Maze.Direction]
        w, h = maze.width, maze.height

        scale = options["scale"]
        wall = maze.wall

        for i in range(w):
            j = h - 1
            if wall(i, j, NORTH):
                self.draw_line(i * scale - 1, (j + 1) * scale,
                               (i + 1) * scale, (j + 1) * scale,
                               Color.FOREGROUND)
            j = 0
            if wall(i, j, SOUTH):
                self.draw_line(i * scale - 1, j * scale - 1,
                               (i + 1) * scale, j * scale - 1,
                               Color.FOREGROUND)

        for j in range(h):
            i = w - 1
            if wall(i, j, EAST):
                self.draw_line((i + 1) * scale, j * scale - 1,
                               (i + 1) * scale, (j + 1) * scale,
                               Color.FOREGROUND)
            i = 0
            if wall(i, j, WEST):
                self.draw_line(i * scale - 1, j * scale - 1,
                               i * scale - 1, (j + 1) * scale,
                               Color.FOREGROUND)

        for i in range(w):
            for j in range(h):
                if wall(i, j, EAST):
                    self.draw_line((i + 1) * scale - 1, j * scale - 1,
                                   (i + 1) * scale - 1, (j + 1) * scale,
                                   Color.FOREGROUND)
                if wall(i, j, NORTH):
                    self.draw_line(i * scale - 1, (j + 1) * scale - 1,
                                   (i + 1) * scale, (j + 1) * scale - 1,
                                   Color.FOREGROUND)
                if wall(i, j, WEST):
                    self.draw_line(i * scale, j * scale - 1,
                                   i * scale, (j + 1) * scale,
                                   Color.FOREGROUND)
                if wall(i, j, SOUTH):
                    self.draw_line(i * scale - 1, j * scale,
                                   (i + 1) * scale, j * scale,
                                   Color.FOREGROUND)

        if options["solution"] and maze.solution is not None:
            if maze.unicursive():
                ignored_cells = {(i_, j_) for i_ in range(w) for j_ in range(h)}
                for e in maze.solution:
                    ignored_cells.remove(e)
                for i, j in ignored_cells:
                    self.fill_rect(i * scale, j * scale, scale, scale,
                                   Color.FOREGROUND, Color.FOREGROUND)

            # painter.setPen(Qt.blue)
            i0, j0 = maze.solution[0]
            # painter.drawRect(QRectF((i0 + .5) * scale - 1, (j0 + .5) * scale - 1, 2, 2))

            for i, (i1, j1) in enumerate(maze.solution[1:]):
                self.draw_line((i0 + .5) * scale, (j0 + .5) * scale - 1,
                               (i1 + .5) * scale, (j1 + .5) * scale - 1,
                               Color.PATH)
                i0, j0 = i1, j1

        for t in SignType:
            positions = maze.signs_data[t]
            images = options[t.value.lower() + "s"]
            if images and positions is not None:
                for vix, six, d, _ in positions:
                    c_i, c_j = maze.solution[six]
                    img = images[vix][d.value]
                    self.draw_image(c_i * scale + 1, c_j * scale + 1,
                                    scale - 2, scale - 2,
                                    img)

        for f, (i, j), c in [(self.draw_start, maze.start, Color.START),
                             (self.draw_finish, maze.end, Color.FINISH)]:
            f(i * scale, j * scale, scale, scale, c)

        if r := options.get("robot"):
            i, j = r["pos"]
            if options["outputs"] is OutputType.DISCRETE:
                self.fill_rect((i - .5) * scale, (j - .5) * scale,
                               scale-1, scale-1,
                               Color.COLOR1, Color.COLOR1, alpha=.25)

            elif options["outputs"] is OutputType.CONTINUOUS:
                x, y = scale * r["pos"]
                self.fill_circle(x, y, Robot.RADIUS * scale,
                                 None, Color.COLOR1)

                if v := r.get("vel"):
                    v *= scale
                    self.draw_line(x, y, x+v.x, y+v.y, Color.COLOR3)

                if a := r.get("acc"):
                    a *= scale
                    self.draw_line(x, y, x+a.x, y+a.y, Color.COLOR1)

            else:
                raise ValueError
