from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox)

import amaze


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(640, 480)

        # Create horizontal layout
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Create dedicated widget for rendering mazes
        self.maze_widget = amaze.MazeWidget(
            self._maze_data(0, 5, 0, True,
                            amaze.StartLocation.SOUTH_WEST))
        layout.addWidget(self.maze_widget)

        # Build content
        self.widgets, sub_layout = self._create_widgets()
        layout.addLayout(sub_layout)

    def _create_widgets(self):
        # Create a secondary vertical layout for inputs
        sub_layout = QFormLayout()

        widgets = {}

        def _add(cls, name, signal, func=None):
            sub_layout.addRow(name, _w := cls())
            if func is not None:
                func(_w)
            getattr(_w, signal).connect(self.reset_maze)
            widgets[name] = _w

        _add(QSpinBox, "Seed", "valueChanged")

        _add(QSpinBox, "Size", "valueChanged",
             lambda w: w.setRange(5, 20))

        _add(QDoubleSpinBox, "Lures", "valueChanged",
             lambda w: (
                 w.setRange(0, 100),
                 w.setSuffix("%")
             ))

        _add(QCheckBox, "Unicursive", "clicked",
             lambda w: w.setChecked(True))

        _add(QComboBox, "Start", "currentTextChanged",
             lambda w: w.addItems([s.name for s in amaze.StartLocation]))

        return widgets, sub_layout

    def reset_maze(self):
        self.maze_widget.set_maze(self._maze_data(
            self.widgets["Seed"].value(),
            self.widgets["Size"].value(),
            self.widgets["Lures"].value() / 100,
            self.widgets["Unicursive"].isChecked(),
            amaze.StartLocation[self.widgets["Start"].currentText()]
        ))

    @staticmethod
    def _maze_data(seed, size, p_lure, easy, start):
        return amaze.Maze.BuildData(
            seed=seed,
            width=size, height=size,
            unicursive=easy,
            rotated=True,
            start=start,
            clue=[amaze.Sign(value=1)],
            p_lure=p_lure,
            lure=[amaze.Sign(value=.5)],
            p_trap=0,
            trap=[]
        )


def main(is_test=False):
    # Create main QT objects
    app = amaze.application()
    window = MainWindow()
    window.show()

    if is_test:
        QTimer.singleShot(1000, lambda: window.close())

    # Run
    app.exec()


if __name__ == "__main__":
    main()
