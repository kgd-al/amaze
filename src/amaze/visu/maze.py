from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QSize, QRectF, QPointF
from PyQt5.QtGui import QPainter, QImage, QPainterPath
from PyQt5.QtWidgets import QLabel

from amaze.simu.simulation import Simulation, OutputType
from amaze.visu import resources


class MazeWidget(QLabel):
    def __init__(self):
        super().__init__("Generate a maze\n"
                         "using controls\n"
                         "on the right >>>")
        self._simulation: Optional[Simulation] = None
        self._scale = 12
        self._solution = True

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

    def draw_to(self, path):
        img = QImage(self._scale * self._simulation.maze.width + 2,
                     self._scale * self._simulation.maze.height + 2,
                     QImage.Format_RGB32)
        painter = QPainter(img)

        img.fill(Qt.white)

        self._render(painter)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if img.save(str(path)):
            print("Wrote to", path)
        else:
            print("Error writing to", path)

    def _render(self, painter: QPainter):
        maze = self._simulation.maze
        D = maze.Direction
        w, h = maze.width, maze.height

        scale = self._scale

        wall = maze.wall

        text_debug = False
        if not text_debug:
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
            if text_debug:
                painter.setPen(Qt.gray)
                for i in range(w):
                    for j in range(h):
                        painter.drawText(QRectF(i*scale, j*scale, scale, scale),
                                         Qt.AlignCenter, f"{i}, {j}")

            painter.setPen(Qt.blue)
            i0, j0 = maze.solution[0]
            painter.drawRect(QRectF((i0+.5)*scale-1, (j0+.5)*scale-1, 2, 2))

            for i, (i1, j1) in enumerate(maze.solution[1:]):
                painter.drawLine(QPointF((i0+.5)*scale, (j0+.5)*scale-1),
                                 QPointF((i1+.5)*scale, (j1+.5)*scale-1))

                if text_debug:
                    il, ir = min(i0, i1), max(i0, i1)
                    jl, jr = min(j0, j1), max(j0, j1)
                    tr = QRectF(il * scale, jl * scale,
                                (ir - il + 1) * scale,
                                (jr - jl + 1) * scale)
                    painter.drawText(tr, Qt.AlignCenter, str(i))

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

        if (p := self._simulation.robot.pos) is not None:
            i, j = p
            if self._simulation.data.outputs == OutputType.DISCRETE:
                painter.setPen(Qt.red)
                painter.drawRect(QRectF((i-.5)*scale, (j-.5)*scale, scale, scale))

            elif self._simulation.data.outputs == OutputType.CONTINUOUS:
                path = QPainterPath()
                path.addEllipse(QRectF(-.5*scale, -.5*scale, scale, scale))
                painter.translate(scale * QPointF(*p))
                painter.fillPath(path, Qt.red)

            else:
                raise ValueError

        painter.end()
