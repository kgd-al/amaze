from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QLabel


class TinyLabel(QLabel):
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return 0


class InputsLabel(TinyLabel):
    def heightForWidth(self, w: int) -> int:
        return w if self.pixmap() else 0

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
    pass


class ValuesLabel(TinyLabel):
    pass
