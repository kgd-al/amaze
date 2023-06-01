from logging import getLogger
from typing import Optional, Union

import numpy as np
from PyQt5.QtCore import QSettings, QTimer, QRectF, QSize, QEvent, QBuffer, QIODevice, Qt
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QHelpEvent
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout, QToolButton, QSpinBox, \
    QGroupBox, QStyle, QAbstractButton, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QScrollArea, QToolTip

from amaze.simu.env.maze import Maze
from amaze.simu.simulation import Simulation, Robot, InputType, OutputType, ControlType
from amaze.visu import resources
from amaze.visu.maze import MazeWidget
from amaze.visu.widgets.combobox import ZoomingComboBox
from amaze.visu.widgets.labels import InputsLabel, OutputsLabel, ValuesLabel

logger = getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, args: Optional = None):
        super().__init__()

        holder = QWidget()
        layout = QHBoxLayout()

        self.buttons: dict[str, QAbstractButton] = {}
        self.config, self.stats = {}, {}
        self.visu: dict[str, QLabel] = {}

        self.maze_w = MazeWidget()
        layout.addWidget(self.maze_w)
        layout.setStretch(0, 1)

        layout.addWidget(self._build_controls())
        layout.setStretch(1, 0)

        holder.setLayout(layout)
        self.setCentralWidget(holder)

        self.playing = False
        self.speed = 1
        self.timer_dt = .1

        self.timer = QTimer()

        self.simulation = Simulation()
        self.maze_w.set_simulation(self.simulation)
        self.timer.timeout.connect(self._step)

        if args is not None:
            self._restore_settings(args)

        self._connect_signals()

        self.buttons["play"].setFocus()

    def _maze_data(self):
        def _sbv(name): return self.config[name].value()
        def _cbv(name): return self.config[name].currentText()
        def _cbb(name): return self.config[name].isChecked()
        def _on(name): return _cbb("with_" + name + "s")
        return Maze.BuildData(
            width=_sbv("width"), height=_sbv("height"),
            seed=_sbv("seed"),
            unicursive=_cbb("unicursive"),
            cue=_cbv("cue") if _on("cue") else None,
            p_trap=float(_sbv("p_trap")),
            trap=_cbv("trap") if _on("trap") else None
        )

    def _robot_data(self):
        def _sbv(name): return self.config[name].value()

        def _cbv(name, enum):
            return enum[self.config[name].currentText().upper()]

        return Robot.BuildData(
            vision=_sbv("vision"),
            inputs=_cbv("inputs", InputType),
            outputs=_cbv("outputs", OutputType),
            control=_cbv("control", ControlType),
        )

    def generate_maze(self):
        self.reset()

        maze = self.simulation.maze
        self.stats["m_size"].setText(
            f"{maze.width} x {maze.height} ({maze.width * maze.height} cells)")
        self.stats["m_path"].setText(str(len(maze.solution)))
        self.stats["m_intersections"].setText(str(len(maze.intersections)))
        self.stats["m_traps"].setText(
            str(len(maze.traps)) if maze.traps is not None else "/")

    def start(self):
        print("start")
        self._play(True)

    def pause(self):
        print("pause")
        self._play(False)

    def next(self):
        self._step()

    def stop(self):
        self._play(False)
        self.reset()

    def reset(self):
        self.simulation.reset(Maze.generate(self._maze_data()),
                              self._robot_data())
        self.maze_w.set_resolution(self.simulation.data.vision)
        self.simulation.generate_inputs()
        self.gb_config.setEnabled(True)
        self._update()

    def _play(self, play: Optional[bool] = None):
        if play is None:
            self.playing = not self.playing
        else:
            self.playing = play

        if self.playing:
            icon = QStyle.SP_MediaPause
            self.timer.start(round(1000 * self.timer_dt))
        else:
            icon = QStyle.SP_MediaPlay
            self.timer.stop()

        self.gb_config.setEnabled(False)

        self.buttons["play"].setIcon(self.style().standardIcon(icon))

    def _step(self):
        self.simulation.step()
        self._update()
        if self.simulation.done():
            self.stop()

    def _update(self):
        def update(k, v, fmt="{}"): self.stats[k].setText(fmt.format(v))
        update("s_step", f"{self.simulation.time()} "
                         f"({self.simulation.timestep})")
        update("r_pos", self.simulation.robot.pos)
        update("r_reward", self.simulation.robot.reward, "{:g}")

        if self.simulation.data is not None:
            s = self.simulation.data.vision
            def scale(x): return int(255*x)
            pi = np.vectorize(scale)(self.simulation.robot.inputs)\
                .astype('uint8').copy()
            i = QImage(pi, s, s, s, QImage.Format_Grayscale8)
            self.visu["img_inputs"].setPixmap(QPixmap.fromImage(i))

        self.maze_w.update()

    def _build_controls(self):
        def container(name, contents):
            c = QGroupBox(name)
            c.setLayout(contents)
            return c

        holder = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(container("Robot", self._build_robot_observers()))
        layout.addWidget(container("Render", self._build_rendering_config()))

        self.gb_config = config = \
            container("Config", self._build_config_controls())
        layout.addWidget(config)
        layout.addStretch(1)

        layout.addWidget(container("Statistics", self._build_stats_labels()))

        layout.addWidget(container("Controls", self._build_control_buttons()))

        holder.setLayout(layout)
        return holder

    def _build_robot_observers(self):
        def widget(k, t):
            w_ = t()
            self.visu["img_" + k.lower()] = w_
            return w_

        layout = QVBoxLayout()
        for n, d, t in [("Inputs", "(t)", InputsLabel),
                        ("Outputs", "(t-1)", OutputsLabel),
                        ("Values", "(t-1)", ValuesLabel)]:
            layout.addLayout(self._section(n + " " + d))
            layout.addWidget(widget(n, t))

        return layout

    def _build_rendering_config(self):
        layout = QVBoxLayout()

        def widget(k, t, *args):
            w = t(*args)
            self.config[k] = w
            layout.addWidget(w)
            return w

        widget("solution", QCheckBox, "Show solution")

        return layout

    def _build_config_controls(self):
        layout = QVBoxLayout()

        def widget(cls, name, **kwargs):
            w_ = cls(**kwargs)
            self.config[name] = w_
            return w_

        def row(label: str, widgets: Union[list[QWidget], QWidget]):
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(QLabel(label))
            if not isinstance(widgets, list):
                widgets = [widgets]
            for w_ in widgets:
                sub_layout.addWidget(w_)
            layout.addLayout(sub_layout)
            return sub_layout

        size_layout = \
            row("Size", [widget(QSpinBox, "width"),
                         QLabel("x"),
                         widget(QSpinBox, "height")])
        for i, s in enumerate([0, 1, 0, 1]):
            size_layout.setStretch(i, s)
        for w in ["width", "height"]:
            self.config[w].setRange(2, 100)

        w = widget(QSpinBox, "seed")
        row("Seed", w)
        w.setRange(0, 2**31-1)
        w.setWrapping(True)

        w = widget(QCheckBox, "unicursive")
        row("Easy", w)

        cues = QGroupBox("Cues")
        cues.setCheckable(True)
        self.config["with_cues"] = cues
        cues_layout = QVBoxLayout()
        cues_layout.addWidget(widget(ZoomingComboBox, "cue"))
        cues.setLayout(cues_layout)
        layout.addWidget(cues)

        traps = QGroupBox("Trap")
        traps.setCheckable(True)
        self.config["with_traps"] = traps
        traps_layout = QHBoxLayout()
        w = widget(QDoubleSpinBox, "p_trap")
        traps_layout.addWidget(w)
        w.setSuffix('%')
        w.setRange(.01, 100)
        traps_layout.addWidget(widget(QComboBox, "trap"))
        traps.setLayout(traps_layout)
        layout.addWidget(traps)

        for k in ["cue", "trap"]:
            cb: QComboBox = self.config[k]
            for k_, v in resources.resources():
                cb.addItem(QIcon(QPixmap.fromImage(v)), k_)

        layout.addLayout(self._section("Robot"))
        w = widget(QSpinBox, "vision")
        row("Vision", w)
        w.setRange(5, 31)
        w.setSingleStep(2)

        cb = widget(QComboBox, "inputs")
        cb.addItems([v.name.lower() for v in InputType])
        row("Inputs", cb)

        cb = widget(QComboBox, "outputs")
        cb.addItems([v.name.lower() for v in OutputType])
        row("Outputs", cb)

        cb = widget(QComboBox, "control")
        cb.addItems([v.name.lower() for v in ControlType])
        row("Control", cb)

        return layout

    def _build_stats_labels(self):
        layout = QFormLayout()

        def add_fields(names):
            for name in names:
                lbl = QLabel()
                layout.addRow(name[2].upper() + name[3:].lower(), lbl)
                self.stats[name] = lbl

        layout.addRow(self._section("Maze"))
        add_fields(["m_size", "m_path", "m_intersections", "m_traps"])
        layout.addRow(self._section("Simulation"))
        add_fields(["s_step"])
        layout.addRow(self._section("Robot"))
        add_fields(["r_pos", "r_reward"])

        return layout

    def _build_control_buttons(self):
        layout = QHBoxLayout()
        b = self.__button
        sp = QStyle.StandardPixmap
        b(layout, "slow", sp.SP_MediaSeekBackward)
        b(layout, "stop", sp.SP_MediaStop, "Ctrl+Esc")
        b(layout, "play", sp.SP_MediaPlay, "Ctrl+Pause")
        b(layout, "next", sp.SP_ArrowForward, "Ctrl+N")
        b(layout, "fast", sp.SP_MediaSeekForward)
        return layout

    @staticmethod
    def _section(name_):
        def frame():
            f = QFrame()
            f.setFrameStyle(QFrame.HLine | QFrame.Raised)
            return f

        layout = QHBoxLayout()
        layout.addWidget(frame())
        layout.addWidget(QLabel(name_))
        layout.addWidget(frame())
        for i, s in enumerate([1, 0, 1]):
            layout.setStretch(i, s)

        return layout

    def __button(self, layout, name, icon, shortcut = None):
        button = QToolButton()
        button.setIcon(self.style().standardIcon(icon))
        if shortcut is not None:
            button.setShortcut(shortcut)
        layout.addWidget(button)
        self.buttons[name] = button
        return button

    def _connect_signals(self):
        for k in ["width", "height", "seed", "vision", "p_trap"]:
            self.config[k].valueChanged.connect(self.generate_maze)
        for k in ["trap", "cue"]:
            self.config[k].currentTextChanged.connect(self.generate_maze)
            self.config["with_" + k + "s"].clicked.connect(self.generate_maze)
        self.config["unicursive"].clicked.connect(self.generate_maze)

        self.config["solution"].clicked.connect(self.maze_w.show_solution)

        def connect(name, f): self.buttons[name].clicked.connect(f)
        connect("stop", self.stop)
        connect("play", lambda: self._play(None))
        connect("next", self.next)

    # =========================================================================

    @staticmethod
    def _settings():
        return QSettings("kgd", "amaze")

    def _restore_settings(self, args):
        config = self._settings()
        logger.info(f"Loading configuration from {config.fileName()}")

        def _try(name, f, default=None):
            if (v := config.value(name)) is not None:
                return f(v)
            else:
                return default
        _try("pos", lambda p: self.move(p))
        _try("size", lambda s: self.resize(s))

        def restore_show_solution(s):
            s = bool(int(s))
            self.config["solution"].setChecked(s)
            self.maze_w.show_solution(s)

        _try("solution", restore_show_solution)

        self._restore_from_robot_build_data(
            Robot.BuildData(**_try("robot", lambda v: v, dict())),
            Robot.BuildData.from_argparse(args))

        self._restore_from_maze_build_data(
            Maze.BuildData(**_try("maze", lambda v: v, dict())),
            Maze.BuildData.from_argparse(args))

    def _save_settings(self):
        config = self._settings()
        config.setValue('pos', self.pos())
        config.setValue('size', self.size())
        config.setValue('solution', int(self.config["solution"].isChecked()))
        config.setValue('maze', self._maze_data().__dict__)
        config.setValue('robot', self._robot_data().__dict__)
        config.sync()

    def closeEvent(self, _):
        self._save_settings()

    # =========================================================================

    def _restore_from_maze_build_data(self,
                                      save_data: Maze.BuildData,
                                      argv_data: Maze.BuildData):
        print(f"[0] {save_data=}")
        print(f"[1] {argv_data=}")

        def val(k):
            return v if (v := getattr(argv_data, k)) is not None \
                else getattr(save_data, k)

        for name in ["width", "height", "seed", "p_trap"]:
            if v := val(name):
                self.config[name].setValue(v)

        for name in ["cue", "trap"]:
            self.config[name].setCurrentText(val(name))

        for name, val in [("with_cues", val("cue") is not None),
                          ("with_traps", val("trap") is not None),
                          ("unicursive", val("unicursive"))]:
            self.config[name].setChecked(val)

    def _restore_from_robot_build_data(self,
                                       save_data: Robot.BuildData,
                                       argv_data: Robot.BuildData):

        def val(k):
            return v if (v := getattr(argv_data, k)) is not None \
                else getattr(save_data, k)

        for name in ["vision"]:
            self.config[name].setValue(val(name))

        for name in ["inputs", "outputs", "control"]:
            self.config[name].setCurrentText(val(name).name.lower())
