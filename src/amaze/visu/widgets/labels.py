from logging import getLogger

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QRectF, Qt, QPointF
from PyQt5.QtGui import QPainter, QImage, QPixmap, QColor
from PyQt5.QtWidgets import QLabel

from amaze.simu.controllers.base import BaseController
from amaze.simu.types import InputType, OutputType, Action, State

logger = getLogger(__name__)


class TinyLabel(QLabel):
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return 0


class InputsLabel(TinyLabel):
    def heightForWidth(self, w: int) -> int:
        return w if self.pixmap() else 0

    def set_inputs(self, inputs: np.ndarray, i_type: InputType):
        if i_type == InputType.CONTINUOUS:
            def scale(x): return int(255 * x)
            w, h = inputs.shape
            pi = np.vectorize(scale)(inputs).astype('uint8').copy()
            i = QImage(pi, h, w, h, QImage.Format_Grayscale8)
            self.setPixmap(QPixmap.fromImage(i))

        else:
            coordinates = [
                (3, 2), (2, 1), (1, 2), (2, 3,), (4, 2), (2, 0), (0, 2), (2, 4)
            ]
            colors = {0: Qt.black, 1: Qt.white, .5: Qt.red}

            img = QImage(5, 5, QImage.Format_ARGB32)
            img.fill(Qt.transparent)
            for i, (v, pos) in enumerate(zip(inputs, coordinates)):
                c = colors[v] if i < 4 else QColor.fromHslF(0, 0, v)
                img.setPixelColor(*pos, c)
            self.setPixmap(QPixmap.fromImage(img))

    def paintEvent(self, _) -> None:
        p = self.pixmap()
        if self.pixmap() is None:
            return
        painter = QPainter(self)
        sw, sh = self.width(), self.height()
        pw, ph = p.width(), p.height()
        scale = min(sw / pw, sh / ph)
        painter.drawPixmap(QRectF(.5 * (sw - scale * pw),
                                  .5 * (sh - scale * ph),
                                  scale * pw, scale * ph).toAlignedRect(),
                           p, p.rect())


class OutputsLabel(TinyLabel):
    def __init__(self):
        super().__init__()
        self.action = None
        self.output_type = None

    def heightForWidth(self, w: int) -> int:
        return w

    def set_outputs(self, outputs: Action, o_type: OutputType):
        self.action = tuple(outputs)
        self.output_type = o_type
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        s = min(w, h)

        pen = painter.pen()
        pen.setWidthF(0)
        painter.setPen(pen)

        if self.output_type is OutputType.DISCRETE:
            painter.translate(.5 * (w - s), .5 * (h - s))
            painter.scale(s/3-1, s/3-1)
            painter.drawRect(1, 0, 1, 3)
            painter.drawRect(0, 1, 3, 1)
            p = {(+1, 0): (2, 1), (0, +1): (1, 0),
                 (-1, 0): (0, 1), (0, -1): (1, 2),
                 (0, 0): (1, 1)}
            if self.action:
                painter.fillRect(QRectF(*p[self.action], 1, 1), Qt.white)
        else:
            painter.translate(.5 * w, .5 * h)
            painter.scale(.5*s, -.5*s)
            painter.drawLine(-1, 0, 1, 0)
            painter.drawLine(0, -1, 0, 1)
            pen.setColor(Qt.red)
            painter.setPen(pen)
            if self.action:
                painter.drawLine(QPointF(0, 0),
                                 QPointF(*self.action))


class ValuesLabel(TinyLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.values = []

    def heightForWidth(self, w: int) -> int:
        return w

    def set_values(self, c: BaseController, s: State):
        value_f = getattr(c, 'values', None)
        if value_f is None:
            self.setText("N/A")
            return
        self.setText("")
        self.values = value_f(s)
        # l, u = c.min_value, c.max_value
        l, u = min(self.values), max(self.values)
        # r = max(abs(l), abs(u))
        # l, u = -r, r
        d = u-l
        if d != 0:
            self.values = [(v, 2 * ((v-l)/d) - 1) for v in self.values]
        else:
            self.values = [(v, v) for v in self.values]
        self.update()

    def paintEvent(self, e) -> None:
        if self.text():
            super().paintEvent(e)
            return
        painter = QPainter(self)
        w, h = self.width(), self.height()
        size = min(w, h)
        s = size/3
        dx, dy = .5 * (w - size), .5 * (h - size)
        painter.translate(dx, dy)
        for p, (v, u) in zip([(2, 1), (1, 0), (0, 1), (1, 2)], self.values):
            c = QColor.fromRgbF(
                -u if u < 0 else 0,
                u if u > 0 else 0,
                0
            )
            r = QRectF(p[0]*s, p[1]*s, 1*s, 1*s)
            painter.setPen(c)
            painter.fillRect(r, c)
            painter.setPen(Qt.white)
            painter.drawText(r, Qt.AlignCenter, f"{v:+.2g}")


class ElidedLabel(QLabel):
    def __init__(self, mode=Qt.ElideNone, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elide_mode = mode
        self._cachedText = ""
        self._cachedElidedText = ""

    def set_elide_mode(self, mode: Qt.TextElideMode):
        self._elide_mode = mode
        self._cachedText = ""
        self.update()

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        self._cachedText = ""

    def paintEvent(self, e: QtGui.QPaintEvent):
        if self._elide_mode == Qt.ElideNone:
            return super().paintEvent(e)

        self._update_cached_texts()
        super().setText(self._cachedElidedText)
        super().paintEvent(e)
        super().setText(self._cachedText)

    def _update_cached_texts(self):
        txt = self.text()
        if txt == self._cachedText:
            return
        self._cachedText = txt
        fm = self.fontMetrics()
        self._cachedElidedText = fm.elidedText(
            txt, self._elide_mode, self.width(), Qt.TextShowMnemonic
        )
