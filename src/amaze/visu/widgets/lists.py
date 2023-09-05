from typing import Optional

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QSizePolicy, QListWidget, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox, \
    QToolButton, QLayout

from amaze.visu import resources
from amaze.visu.resources import Sign, Signs


class SignList(QWidget):
    dataChanged = pyqtSignal()

    def __init__(self, controls: Optional[dict[str, QWidget]] = None):
        super().__init__()
        outer_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        inner_layout = QVBoxLayout()
        self.inner_layout = inner_layout

        outer_layout.addLayout(top_layout)
        outer_layout.addLayout(inner_layout)

        if controls:
            for w in controls.values():
                top_layout.addWidget(w)
        top_layout.addStretch()
        b_add = QToolButton()
        b_add.setIcon(QIcon.fromTheme("list-add"))
        top_layout.addWidget(b_add)

        self.setLayout(outer_layout)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        b_add.clicked.connect(lambda: self.add_row())

        self._original_max_height = self.maximumHeight()

    def count(self):
        return self.inner_layout.count()

    def signs(self):
        return [self._sign(self.inner_layout.itemAt(i).widget())
                for i in range(self.inner_layout.count())]

    def set_signs(self, signs: Signs):
        while self.count():
            self.remove_row(self.inner_layout.itemAt(0).widget())
        for sign in signs:
            self.add_row(sign)

    @staticmethod
    def _sign(w: QWidget):
        assert isinstance(w, QWidget)
        assert isinstance(w.layout(), QLayout)
        cb, sb = w.layout().itemAt(0).widget(), w.layout().itemAt(1).widget()
        assert isinstance(cb, QComboBox) and isinstance(sb, QDoubleSpinBox)
        return Sign(cb.currentText(), sb.value())

    def hide(self, hide: bool = True):
        # self.setMaximumHeight(0 if hide else self._original_max_height)
        self.setVisible(not hide)

    def add_row(self, sign=Sign()):
        w = QWidget()
        w.setContentsMargins(0, 0, 0, 0)
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        cb = QComboBox()
        cb.addItems(resources.names())
        cb.setCurrentText(sign.name)
        layout.addWidget(cb, 2)
        sb = QDoubleSpinBox()
        sb.setRange(0, 1)
        sb.setSingleStep(.05)
        sb.setValue(sign.value)
        layout.addWidget(sb, 1)
        tb = QToolButton()
        tb.setIcon(QIcon.fromTheme("edit-delete"))
        layout.addWidget(tb, 0)
        w.setLayout(layout)
        self.inner_layout.addWidget(w)

        self.dataChanged.emit()

        tb.clicked.connect(lambda: self.remove_row(w))
        cb.currentTextChanged.connect(self.dataChanged)
        sb.valueChanged.connect(self.dataChanged)

    def remove_row(self, source: QWidget):
        self.inner_layout.removeWidget(source)
        source.setParent(None)
        self.dataChanged.emit()


class CompactList(QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizeAdjustPolicy(self.AdjustToContents)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)

    def minimumSizeHint(self): return QSize(0, 0)

    def viewportSizeHint(self):
        if self.sizeAdjustPolicy() != self.AdjustToContents:
            return super().viewportSizeHint()
        if not self.model() or self.model().rowCount() == 0:
            return QSize(0, 0)
        return super().viewportSizeHint()

