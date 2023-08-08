from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QToolButton, QWidget, QGroupBox


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(
            text=title, checkable=True, checked=False
        )
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(Qt.RightArrow)
        # self.toggle_button.pressed.connect(self.on_pressed)
        self.toggle_button.toggled.connect(self.on_pressed)

        self.content_area = QGroupBox()
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self._maximum_height = self.content_area.maximumHeight()

        lay = QVBoxLayout()
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)
        self._set_layout(lay)

    @pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.DownArrow if not checked else Qt.RightArrow
        )
        self.content_area.setMaximumHeight(
            0 if checked else 1000
        )

    def collapsed(self):
        return self.toggle_button.isChecked()

    def set_collapsed(self, collapsed):
        self.toggle_button.setChecked(collapsed)

    def setEnabled(self, enabled: bool, child_only: bool = True):
        if child_only:
            self.content_area.setEnabled(enabled)
        else:
            super().setEnabled(enabled)

    def _set_layout(self, layout):
        super().setLayout(layout)

    def setLayout(self, layout):
        self.content_area.setLayout(layout)
