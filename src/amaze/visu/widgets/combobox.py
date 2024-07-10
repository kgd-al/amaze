from PyQt5.QtCore import QEvent, QBuffer, QIODevice
from PyQt5.QtGui import QHelpEvent
from PyQt5.QtWidgets import QComboBox, QToolTip

from ...misc import resources, Sign


class ZoomingComboBox(QComboBox):
    def __init__(self, *args, value_getter, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_getter = value_getter

    def event(self, e: QEvent):
        if e.type() == QEvent.ToolTip and (name := self.currentText()):
            value = self.value_getter()
            color = "white" if (value < .5 or name not in resources.builtins()) else "black"
            i = resources.image(Sign(name, value), 128)
            e: QHelpEvent

            p = i.scaledToHeight(128)
            buffer = QBuffer()
            buffer.open(QIODevice.WriteOnly)
            p.save(buffer, "png")
            image = bytes(buffer.data().toBase64()).decode()
            html = f'<img src="data:image/png;base64,{image}" style="background-color:{color};">'
            QToolTip.showText(e.globalPos(), html)
            return True

        return super().event(e)
