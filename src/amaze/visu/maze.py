from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Optional, List, Tuple, Callable

from PyQt5.QtCore import Qt, QSize, QRectF, QPointF, QLineF
from PyQt5.QtGui import QPainter, QImage, QPainterPath, QColor, QLinearGradient
from PyQt5.QtWidgets import QLabel

from amaze.simu.robot import OutputType, InputType, Robot
from amaze.simu.simulation import Simulation
from amaze.visu import resources

logger = getLogger(__name__)


class MazeWidget(QLabel):
    def __init__(self):
        super().__init__("Generate a maze\n"
                         "using controls\n"
                         "on the right >>>")
        self._simulation: Optional[Simulation] = None
        self._scale = 12
        self._solution = True
        self._robot = True

        self._vision = None
        self._cues = None
        self._traps = None

    def set_simulation(self, simulation: Simulation):
        self._simulation = simulation
        self.update()

    def set_resolution(self, r: int):
        self._vision = r
        
        def _set(v):
            return resources.qt_images(v, r - 2) if v is not None else None
        maze = self._simulation.maze
        self._cues = _set(maze.cue)
        self._traps = _set(maze.trap)

    def show_solution(self, show):
        self._solution = show
        self.update()

    def show_robot(self, show):
        self._robot = show
        self.update()

    def minimumSize(self) -> QSize:
        return 3 * QSize(self._simulation.maze.width,
                         self._simulation.maze.height)

    def paintEvent(self, e):
        maze = self._simulation.maze
        if maze is None:
            super().paintEvent(e)
            return

        self._scale = round(min(self.width() / maze.width,
                                self.height() / maze.height))

        painter = QPainter(self)

        sw = maze.width * self._scale + 2
        sh = maze.height * self._scale + 2
        painter.translate(.5 * (self.width() - sw), .5 * (self.height() - sh))

        painter.fillRect(QRectF(0, 0, sw, sh), Qt.white)

        self._render(painter)

        painter.end()

    def draw_to(self, path,
                post_render: Optional[Callable] = None,
                shape: Optional[Tuple[int, int]] = (0, 0)):
        img = QImage(self._scale * self._simulation.maze.width + 2 + shape[0],
                     self._scale * self._simulation.maze.height + 2 + shape[1],
                     QImage.Format_RGB32)
        painter = QPainter(img)

        img.fill(Qt.white)

        self._render(painter)
        if post_render:
            post_render(painter)

        painter.end()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if img.save(str(path)):
            logger.debug(f"Wrote to {path}")
        else:
            logger.error(f"Error writing to {path}")

    def _render(self, painter: QPainter):
        maze = self._simulation.maze
        D = maze.Direction
        w, h = maze.width, maze.height

        scale = self._scale

        wall = maze.wall

        painter.translate(1, h*scale)
        painter.scale(1, -1)

        painter.setPen(Qt.black)
        for i in range(w):
            j = h - 1
            if wall(i, j, D.NORTH):
                painter.drawLine(i*scale-1, (j+1)*scale,
                                 (i+1)*scale, (j+1)*scale)
            j = 0
            if wall(i, j, D.SOUTH):
                painter.drawLine(i * scale - 1, j * scale - 1,
                                 (i + 1) * scale, j * scale - 1)

        for j in range(h):
            i = w - 1
            if wall(i, j, D.EAST):
                painter.drawLine((i+1)*scale, j*scale-1,
                                 (i+1)*scale, (j+1)*scale)
            i = 0
            if wall(i, j, D.WEST):
                painter.drawLine(i*scale - 1, j*scale-1,
                                 i*scale - 1, (j+1)*scale)

        for i in range(w):
            for j in range(h):
                if wall(i, j, D.EAST):
                    painter.drawLine((i+1)*scale-1, j*scale-1,
                                     (i+1)*scale-1, (j+1)*scale)
                if wall(i, j, D.NORTH):
                    painter.drawLine(i*scale-1, (j+1)*scale-1,
                                     (i+1)*scale, (j+1)*scale-1)
                if wall(i, j, D.WEST):
                    painter.drawLine(i*scale, j*scale-1,
                                     i*scale, (j+1)*scale)
                if wall(i, j, D.SOUTH):
                    painter.drawLine(i*scale-1, j*scale,
                                     (i+1)*scale, j*scale)

        if self._solution and maze.solution is not None:
            if maze.unicursive():
                ignored_cells = {(i_, j_) for i_ in range(w) for j_ in range(h)}
                for e in maze.solution:
                    ignored_cells.remove(e)
                for i, j in ignored_cells:
                    painter.fillRect(QRectF(
                        i*scale, j*scale, scale, scale
                    ), Qt.black)

            painter.setPen(Qt.blue)
            i0, j0 = maze.solution[0]
            painter.drawRect(QRectF((i0+.5)*scale-1, (j0+.5)*scale-1, 2, 2))

            for i, (i1, j1) in enumerate(maze.solution[1:]):
                painter.drawLine(QPointF((i0+.5)*scale, (j0+.5)*scale-1),
                                 QPointF((i1+.5)*scale, (j1+.5)*scale-1))
                i0, j0 = i1, j1

            painter.setPen(Qt.red)
            for images, positions in [(self._cues, maze.intersections),
                                      (self._traps, maze.traps)]:
                if images is not None and positions is not None:
                    for i, d in positions:
                        c_i, c_j = maze.solution[i]
                        pos = QPointF((c_i+.5)*scale, (c_j+.5)*scale-1)
                        px = images[d.value]
                        painter.save()
                        painter.translate(pos)
                        painter.drawImage(
                            QRectF(-.5*scale+2, -.5*scale+2, scale-4, scale-4)
                            .toRect(),
                            px, px.rect())
                        painter.restore()

        path = QPainterPath()
        path.addRect(QRectF(-2, -.4*scale, 3, .8*scale))
        path.moveTo(1, .4*scale)
        path.lineTo(.4*scale, .2*scale)
        path.lineTo(1, 0)
        path.closeSubpath()

        for (i, j), c in [(maze.start, Qt.red), (maze.end, Qt.green)]:
            painter.save()
            painter.setPen(Qt.black)
            painter.translate(i * scale, j * scale)
            painter.translate(.5 * scale, .5 * scale)
            painter.fillPath(path, c)
            painter.drawPath(path)
            painter.restore()

        if self._robot and (r := self._simulation.robot) is not None:
            i, j = r.pos
            if self._simulation.data.outputs == OutputType.DISCRETE:
                painter.setPen(Qt.red)
                r = QRectF((i-.5)*scale, (j-.5)*scale, scale, scale)
                painter.fillRect(r, QColor.fromRgb(255, 0, 0, 31))
                painter.drawRect(r)

            elif self._simulation.data.outputs == OutputType.CONTINUOUS:
                path = QPainterPath()
                radius = Robot.RADIUS * scale
                path.addEllipse(QRectF(-radius, -radius, 2*radius, 2*radius))
                painter.translate(scale * QPointF(*r.pos))
                painter.fillPath(path, Qt.green)

                if True:
                    painter.save()
                    pen = painter.pen()
                    pen.setWidthF(0)
                    pen.setColor(Qt.blue)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(0, 0), QPointF(*(scale*r.vel)))
                    pen.setColor(Qt.red)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(0, 0), QPointF(*(scale*r.acc)))
                    painter.restore()

            else:
                raise ValueError

    def plot_trajectory(self, path: Path,
                        trajectory: List[Tuple[Tuple[float, float],
                                               Tuple[float, float],
                                               float]]):

        _trajectory = []
        prev_scale = self._scale
        s = 100
        m = 10
        self._scale = s

        maze = self._simulation.maze
        if (maze.cue and not self._cues) or (maze.trap and not self._traps):
            if self._simulation.data.inputs == InputType.CONTINUOUS:
                r = self._simulation.data.vision
            else:
                r = 15
            self.set_resolution(r)

        shortened_trajectory = dict(Counter(trajectory))
        min_count = min(shortened_trajectory.values())
        max_count = max(shortened_trajectory.values())
        dif_count = max_count - min_count

        def _draw_trajectory(painter: QPainter):
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

            # painter.scale(self._scale, self._scale)
            for (x_, y_), a, c in _trajectory:
                painter.save()
                painter.translate(x_ * s, y_ * s)
                painter.rotate(a)
                painter.fillPath(arrow, c)
                painter.setPen(QColor(c).darker())
                painter.drawPath(arrow)
                painter.restore()

            painter.setPen(Qt.black)
            maze = self._simulation.maze
            w, h = maze.width, maze.height
            painter.translate(s * w, 0)
            gradient = QLinearGradient(0, 0, 0, 1)
            gradient.setColorAt(0, Qt.green)
            gradient.setColorAt(1, Qt.red)
            gradient.setCoordinateMode(QLinearGradient.ObjectMode)
            cb_box = QRectF(m, m, s/2, s * h - 2 * m)
            painter.fillRect(cb_box, gradient)
            painter.drawRect(cb_box)

            cb_ticks = min(5, max(dif_count, 2))
            for i_ in range(cb_ticks):
                u = i_ / (cb_ticks - 1)
                cb_y = dif_count * u + min_count
                cb_y = round(cb_y)
                x_, y_ = s / 2 + m, u * (s * h - 2 * m) + m
                painter.drawLine(QLineF(x_, y_, x_+10, y_))
                painter.save()
                painter.translate(x_ + 15, y_)
                painter.scale(1, -1)
                painter.drawText(QRectF(0, -20, s / 2 - m - 15, 40),
                                 Qt.AlignCenter, str(cb_y))
                painter.restore()

        rotations = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}

        r_margin = 0
        if len(shortened_trajectory) < len(trajectory):
            def color(count_):
                if dif_count == 0:
                    return Qt.red
                else:
                    v = (count_ - min_count) / dif_count
                    return QColor.fromRgbF(v, 1-v, 0)
            r_margin = s

        else:
            def color(_): return Qt.black

        for ((x, y), (i, j), _), count in shortened_trajectory.items():
            _trajectory.append(((x, y), rotations[int(i), int(j)],
                                color(count)))
        self.draw_to(path, post_render=_draw_trajectory, shape=(r_margin, 0))
        self._scale = prev_scale
