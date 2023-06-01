from PyQt5.QtCore import QEvent, QBuffer, QIODevice
from PyQt5.QtGui import QHelpEvent
from PyQt5.QtWidgets import QComboBox, QToolTip

from amaze.visu import resources


class ZoomingComboBox(QComboBox):
    def event(self, e: QEvent):
        if e.type() == QEvent.ToolTip \
                and (k := self.currentText()):
            i = resources.image(k, 128)
            e: QHelpEvent

            p = i.scaledToHeight(128)
            buffer = QBuffer()
            buffer.open(QIODevice.WriteOnly)
            p.save(buffer, 'png')
            image = bytes(buffer.data().toBase64()).decode()
            html = f'<img src="data:image/png;base64,{image}">'
            QToolTip.showText(e.globalPos(), html)
            return True

        return super().event(e)
