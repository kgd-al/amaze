import functools
import logging
import math
import os
import pprint
from datetime import time
from enum import IntFlag, Enum
from pathlib import Path
from typing import Optional, Union

try:
    import PIL
    logging.getLogger("PIL.PngImagePlugin").propagate = False
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from PyQt5.QtCore import (QSettings, QTimer, Qt, QSignalBlocker, QObject,
                          pyqtSignal, QPoint)
from PyQt5.QtGui import QImage, QRegion, QPainter
from PyQt5.QtWidgets import (QHBoxLayout, QWidget, QLabel,
                             QVBoxLayout, QToolButton, QSpinBox, QGroupBox,
                             QStyle, QAbstractButton, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                             QFileDialog, QSizePolicy, QScrollArea, QMessageBox)

from amaze.simu._maze_metrics import MazeMetrics
from amaze.simu.controllers.control import (controller_factory,
                                            load, check_types)
from amaze.simu.controllers.random import RandomController
from amaze.simu.maze import Maze, StartLocation
from amaze.simu.robot import Robot
from amaze.simu.types import InputType, OutputType
from amaze.simu.simulation import Simulation
from amaze.visu.resources import SignType
from amaze.visu.widgets.collapsible import CollapsibleBox
from amaze.visu.widgets.labels import (InputsLabel, OutputsLabel, ValuesLabel,
                                       ElidedLabel)
from amaze.visu.widgets.lists import SignList
from amaze.visu.widgets.maze import MazeWidget

logger = logging.getLogger(__name__)


class MainWindow(QWidget):
    maze_changed = pyqtSignal()

    class Sections(Enum):
        ROBOT = "Robot"
        RENDER = "Render"
        CONFIG = "Config"
        CONTROLLER = "Controller"
        STATISTICS = "Statistics"
        CONTROLS = "Controls"

    class Reset(IntFlag):
        NONE = 0
        MAZE = 1
        ROBOT = 2
        CONTROL = 4
        ALL = 7

    def __init__(self, args: Optional = None, runnable=True):
        super().__init__()
        self.args = args

        # holder = QWidget()
        self.layout = QHBoxLayout()

        self.runnable = runnable

        if runnable:
            self.playing = False
            self.speed = 1
            self.timer_dt = .1

            self.timer = QTimer()

            self.next_action = None
        self.controller = None

        # ----

        self.sections: dict[MainWindow.Sections, CollapsibleBox] = {}
        self.buttons: dict[str, QAbstractButton] = {}
        self.config, self.stats = {}, {}
        self.visu: dict[str, Union[InputsLabel, OutputsLabel, ValuesLabel]] = \
            {}

        self.save_on_exit = True

        controls = self._build_controls()

        # holder.setLayout(layout)
        # self.setCentralWidget(holder)
        self.setLayout(self.layout)

        maze_w_options = {}
        if args is not None:  # Restore settings and maze/robot configuration
            maze_w_options = self._restore_settings(args)

        self.simulation = Simulation(maze=self._generate_maze(),
                                     robot=self._robot_data(),
                                     save_trajectory=args and args.plot)

        self.maze_w = MazeWidget(self.simulation)
        self.maze_w.update_config(**maze_w_options)

        self.layout.addWidget(self.maze_w)
        self.layout.setStretch(0, 1)

        self.layout.addWidget(controls)
        self.layout.setStretch(1, 0)

        # ----

        self.robot_mode = args and args.is_robot
        if self.robot_mode:  # Prepare robot mode variables
            self._layout_holder = QWidget()
            self._trajectory_plotter = \
                lambda: self.plot_current_trajectory(
                    args.width,
                    Path(f"tmp/trajectories/human"
                         f"_{Maze.bd_to_string(self.maze_data())}"
                         f"_{time().strftime('%Y-%m-%d_%H-%M-%S')}"
                         f".png"),
                    symlink=True,
                    config=dict(cycles=False)
                )

        else:
            self._layout_holder = None
            self._trajectory_plotter = None

        # ----
        if runnable:
            if args:
                self._generate_controller(args.controller)

            self.timer.timeout.connect(self._step)

        self._connect_signals()

        self._update()
        self.buttons["play"].setFocus()

        self._movie = _MovieRecorder(self, args.movie) if args.movie else None

        if self.robot_mode:  # Trimmed down version for the robot
            self._set_robot_mode_layout()

    # =========================================================================
    # == Miscellaneous controllers
    # =========================================================================

    def set_maze(self, bd: Maze.BuildData):
        assert isinstance(bd, Maze.BuildData)
        self._restore_from_maze_build_data(bd)
        self.reset(self.Reset.ALL)

    def save(self):
        folder = Path(f"tmp/autosaves/{self.simulation.maze.seed}/")
        logger.warning(f"Brute force saving everthing in {folder}")
        folder.mkdir(exist_ok=True, parents=True)

        self.maze_w.render_to(str(folder.joinpath("maze.png")))
        self.visu["img_inputs"].grab().save(
            str(folder.joinpath(
                f"inputs_{self.simulation.data.inputs.name.lower()}.png")))

    def plot_current_trajectory(self, width: int, path: Path,
                                config: Optional[dict] = None,
                                symlink: bool = False):
        r = MazeWidget.plot_trajectory(
            simulation=self.simulation, size=width, path=path,
            config=config
        )

        if symlink:
            symlink_path = path.parent.joinpath("last.png")
            symlink_path.unlink(missing_ok=True)
            symlink_path.symlink_to(path.name)

        return r

    # =========================================================================
    # == Public control interface
    # =========================================================================

    def start(self):
        self._play(True)

    def pause(self):
        self._play(False)

    def next(self):
        self._step()

    def stop(self):
        self._play(False)
        self.reset()

    def reset(self, flags: Reset = Reset.ALL):
        reset = MainWindow.Reset
        self.simulation.reset(
            self._generate_maze() if flags & reset.MAZE else None,
            self._robot_data() if flags & reset.ROBOT else None,
        )
        if self.controller and flags & reset.CONTROL:
            self.controller.reset()

        self.maze_w.set_resolution(self.simulation.data.vision)
        self.simulation.generate_inputs()
        self.sections[self.Sections.CONFIG].setEnabled(True)
        if self.runnable:
            self.next_action = self._think()
        self._update()

        maze_metrics = Simulation.compute_metrics(
            self.simulation.maze, self.simulation.data.inputs,
            self.simulation.data.vision)
        s_str = (
            ', '.join([f'{v:.2g}' for v in
                       maze_metrics[MazeMetrics.SURPRISINGNESS].values()]))
        d_str = f"{maze_metrics[MazeMetrics.DECEPTIVENESS]}"
        for m, v in [(MazeMetrics.SURPRISINGNESS, f"{{{s_str}}}"),
                     (MazeMetrics.DECEPTIVENESS, f"{{{d_str}}}"),
                     (MazeMetrics.INSEPARABILITY, None)]:
            if v is None:
                v = f"{maze_metrics[m]:.2g}"
            self.stats[f"m_{m.name.lower()}"].setText(v)

        title = self.simulation.maze.to_string()
        if self.controller:
            title += f" | {self.controller.name()}"
        self.setWindowTitle(title)

    # =========================================================================
    # == Private control
    # =========================================================================

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

        self.buttons["play"].setIcon(self.style().standardIcon(icon))

    def _think(self):
        if isinstance(self.controller, RandomController):
            self.controller.curr_pos = self.simulation.robot.pos
        next_action = self.controller(self.simulation.observations)
        return next_action

    def _step(self):
        self.sections[self.Sections.CONFIG].setEnabled(False)
        reward = self.simulation.step(self.next_action)
        self.next_action = self._think()

        if reward is None:
            return

        if not self.simulation.done() or self._movie:
            self._update()

        if self._movie:
            self._movie.step()

        if self.simulation.done():
            self._done()

    def _done(self):
        reward = self.simulation.robot.reward
        infos = self.simulation.infos()

        if not self.robot_mode:
            print(f"reward = {reward}. Infos:\n{pprint.pformat(infos)}")
        if not self.args.autoquit:
            QMessageBox.information(
                self, "Failed" if infos["failure"] else "Success",
                f"          Actions: {infos['steps']}\n"
                f"Cumulative reward: {reward}"
                f" ({100 * infos['pretty_reward']:3.2f}%)")

        if self._trajectory_plotter is not None:
            self._trajectory_plotter()

        if self._movie:
            self._movie.save()

        self.stop()

        if self.args.autoquit:
            self.close()

        else:
            if self.robot_mode:
                self.start()

    def _update(self):
        self.maze_w.update()

        if not self.runnable:
            return

        def update(k, v, fmt="{}"): self.stats[k].setText(fmt.format(v))
        update("s_step", f"{self.simulation.time():g}s "
                         f"({self.simulation.timestep} timesteps)")
        update("r_pos",
               ", ".join([f"{v:.2g}" for v in self.simulation.robot.pos]))
        update("r_reward",
               f"{self.simulation.last_reward:+g}"
               f" ({self.simulation.robot.reward:g})")

        if self.simulation.data is not None:
            def lbl(k): return self.visu["img_" + k]
            lbl("inputs").set_inputs(self.simulation.observations,
                                     self.simulation.data.inputs)
            lbl("outputs").set_outputs(self.next_action,
                                       self.simulation.data.outputs)
            lbl("values").set_values(self.controller,
                                     self.simulation.observations)

    # =========================================================================
    # == Config collectors
    # =========================================================================

    def _generate_maze(self):
        maze = Maze.generate(self.maze_data())
        stats = maze.stats()
        self.stats["m_size"].setText(
            f"{stats['size']} ({maze.width * maze.height} cells)")
        self.stats["m_path"].setText(str(stats['path']))
        self.stats["m_intersections"].setText(str(stats['intersections']))
        for t in [SignType.LURE, SignType.TRAP]:
            key = t.value.lower() + "s"
            data = stats[key]
            self.stats[f"m_{key}"].setText(
                str(data) if data is not None else "/")

        self.maze_changed.emit()

        return maze

    def _combobox_value(self, name):
        return self.config[name].currentText().upper()

    def _enum_value(self, name, enum): return enum[self._combobox_value(name)]

    def maze_data(self):
        def _sbv(name): return self.config[name].value()
        def _cbv(name): return self._combobox_value(name)
        def _cbb(name): return self.config[name].isChecked()
        def _sign(name): return self.config[name].signs()
        def _on(name): return _cbb("with_" + name + "s")
        return Maze.BuildData(
            width=_sbv("width"), height=_sbv("height"),
            seed=_sbv("seed"),
            unicursive=_cbb("unicursive"),
            rotated=_cbb("rotated"),
            start=StartLocation[_cbv("start").upper()],
            clue=_sign("clues") if _on("clue") else [],
            lure=_sign("lures") if _on("lure") else [],
            p_lure=float(_sbv("p_lure")) / 100,
            trap=_sign("traps") if _on("trap") else [],
            p_trap=float(_sbv("p_trap")) / 100
        )

    def _robot_data(self):
        def _sbv(name): return self.config[name].value()
        def _cbv(name): return self._combobox_value(name)
        def _ecbv(name, enum): return self._enum_value(name, enum)

        return Robot.BuildData(
            vision=_sbv("vision"),
            inputs=_ecbv("inputs", InputType),
            outputs=_ecbv("outputs", OutputType),
            control=_cbv("control"),
        )

    def _generate_controller(self, new_value=None, open_dialog=False):
        c = getattr(self, 'controller', None)
        if c and new_value is None and not open_dialog:
            return

        ccb: QComboBox = self.config['control']

        settings = self._settings()

        old_path = settings.value("controller", None)
        if open_dialog:
            path, _ = QFileDialog.getOpenFileName(
                parent=self, caption="Open controller",
                directory=old_path,
                filter="Controllers (*.ctrl)"
            )
            if not path:
                return
            else:
                new_value = Path(path)

        elif new_value is None and ccb.currentText() == "autonomous":
            new_value = old_path

        error = False
        if isinstance(new_value, Path):
            if new_value.exists():
                c = load(new_value)

                settings.setValue("controller", new_value)
                settings.sync()

                ccb.setCurrentIndex(ccb.count()-1)
            else:
                logger.warning(f"Could not find controller at '{new_value}'.")
                error = True

        elif isinstance(new_value, str):
            ccb.setCurrentText(new_value)

        if error:
            logger.warning(f"Reverting to keyboard controller")
            ccb.setCurrentText("keyboard")

        ct = ccb.currentText()
        if (ct.lower() != "autonomous") or c is None:
            c = controller_factory(
                ct,
                dict(
                    a_type=self._enum_value("outputs", OutputType)
                )
            )

        check_types(c, self._robot_data())

        simple = c.simple
        if not simple:
            if i_t := c.inputs_types()[0]:
                self.config['inputs'].setCurrentText(i_t.name.lower())
            if o_t := c.outputs_types()[0]:
                self.config['outputs'].setCurrentText(o_t.name.lower())

        self.sections[self.Sections.CONTROLLER].setVisible(not simple)
        if not simple:
            self.stats["c_path"].setText(str(new_value))
            l: QFormLayout = self.stats["c_layout"]

            def clear_layout(l_):
                while l_.count() > 0:
                    i = l_.takeAt(0)
                    if i.widget():
                        i.widget().setParent(None)
                    elif i.layout():
                        clear_layout(i.layout())
            clear_layout(l)

            stats = c.details()
            for k, v in stats.items():
                l.addRow(k, QLabel(str(v)))

        self.controller = c
        self.reset(MainWindow.Reset.ROBOT)

    # =========================================================================
    # == Layout/controls setup
    # =========================================================================

    def showEvent(self, e):
        super().showEvent(e)
        if self._movie:
            self._movie.step()

    def _set_robot_mode_layout(self):
        self._layout_holder.setLayout(self.layout)

        layout = QFormLayout()
        self.setLayout(layout)

        layout.addRow(self.visu["img_inputs"])
        layout.addRow("Reward:", self.stats["r_reward"])

        self._generate_controller("keyboard")
        self._play(True)

    def _build_controls(self):
        holder = QWidget()
        scroll = QScrollArea()
        # scroll.setMinimumWidth(250)
        scroll.setFrameStyle(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout()

        def container(key, contents, *args, **kwargs):
            c = CollapsibleBox(key.value, *args, **kwargs)
            c.setLayout(contents)
            self.sections[key] = c
            layout.addWidget(c)
            return c

        container(self.Sections.ROBOT, self._build_robot_observers())
        container(self.Sections.RENDER, self._build_rendering_config())

        container(self.Sections.CONFIG, self._build_config_controls())

        container(self.Sections.CONTROLLER, self._build_controller_label())

        container(self.Sections.STATISTICS, self._build_stats_labels())

        container(self.Sections.CONTROLS, self._build_control_buttons())

        layout.addStretch(1)

        holder.setLayout(layout)
        scroll.setWidget(holder)
        return scroll

    def _build_robot_observers(self):
        def widget(k, t_):
            w_ = t_()
            self.visu["img_" + k.lower()] = w_
            return w_

        layout = QGridLayout()
        for n, t, c in [("Inputs", InputsLabel, (0, 0, 1, 2)),
                        ("Outputs", OutputsLabel, (1, 0, 1, 1)),
                        ("Values", ValuesLabel, (1, 1, 1, 1))]:
            i, j, si, sj = c
            layout.addWidget(self._section(n), 2*i, j, si, sj)
            layout.addWidget(widget(n, t), 2*i+1, j, si, sj)

        return layout

    def _build_rendering_config(self):
        layout = QGridLayout()
        r, c = 0, 0

        def widget(k, t, *args):
            w = t(*args)
            self.config["show_" + k] = w
            nonlocal r, c
            layout.addWidget(w, r, c)

            c += 1
            r += c // 2
            c = c % 2
            return w

        widget("solution", QCheckBox, "Show solution")
        widget("robot", QCheckBox, "Show robot")
        widget("dark", QCheckBox, "Dark")
        widget("colorblind", QCheckBox, "Colorblind")

        return layout

    def _build_config_controls(self):
        layout = QVBoxLayout()

        def widget(cls, name, *args, **kwargs):
            w_ = cls(*args, **kwargs)
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

        rl = QHBoxLayout()
        rl.addWidget(widget(QCheckBox, "unicursive", "Easy"))
        rl.addWidget(widget(QCheckBox, "rotated", "Rotated"))
        layout.addLayout(rl)

        w = widget(QComboBox, "start")
        w.addItems([v.name.lower() for v in StartLocation])
        row("Start", w)

        def sign_section(name, controls=None):
            signs = QGroupBox(name[0].upper() + name[1:].lower())
            signs.setCheckable(True)
            self.config["with_" + name] = signs
            _layout = QVBoxLayout()
            _layout.setSpacing(0)
            _layout.setContentsMargins(0, 0, 0, 0)

            lst = widget(SignList, name, controls)
            lst.setMinimumHeight(0)
            _layout.addWidget(lst)
            signs.setLayout(_layout)
            return signs

        layout.addWidget(sign_section("clues"))

        w = widget(QDoubleSpinBox, "p_lure")
        w.setSuffix('%')
        w.setRange(0, 100)
        layout.addWidget(sign_section("lures", {"p_lure": w}))

        w = widget(QDoubleSpinBox, "p_trap")
        w.setSuffix('%')
        w.setRange(0, 100)
        layout.addWidget(sign_section("traps", {"p_trap": w}))

        layout.addWidget(self._section("Robot"))
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
        cb.addItems([v.lower() for v in ["Random", "Keyboard", "Autonomous"]])
        row("Control", cb)

        return layout

    def _build_controller_label(self):
        layout = QVBoxLayout()
        h_layout = QHBoxLayout()

        self.__button(h_layout, "c_load", QStyle.SP_DirOpenIcon,
                      "Ctrl+Shift+O")
        self.stats["c_path"] = w = ElidedLabel(mode=Qt.ElideLeft)
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        h_layout.addWidget(w)
        layout.addLayout(h_layout)

        self.stats["c_layout"] = l_ = QFormLayout()
        l_.addRow(QLabel("Connect button"))
        l_.addRow(QLabel("Fill with info from controller"))
        l_.addRow(QLabel("Update value widget"))
        layout.addLayout(l_)

        return layout

    def _build_stats_labels(self):
        layout = QFormLayout()

        def add_fields(names):
            for name in names:
                lbl = QLabel()
                layout.addRow(name[2].upper() + name[3:].lower(), lbl)
                self.stats[name] = lbl

        layout.addRow(self._section("Maze"))
        add_fields(["m_size", "m_path", "m_intersections",
                    "m_lures", "m_traps"]
                   + [f"m_{m.name.lower()}" for m in MazeMetrics])
        layout.addRow(self._section("Simulation"))
        add_fields(["s_step"])
        layout.addRow(self._section("Robot"))
        add_fields(["r_pos", "r_reward"])

        return layout

    def _build_control_buttons(self):
        layout = QHBoxLayout()
        b = self.__button
        sp = QStyle.StandardPixmap
        b(layout, "save", sp.SP_FileIcon, "Ctrl+P")
        b(layout, "slow", sp.SP_MediaSeekBackward)
        b(layout, "stop", sp.SP_MediaStop, "Esc")
        b(layout, "play", sp.SP_MediaPlay, "Pause")
        b(layout, "next", sp.SP_ArrowForward, "N")
        b(layout, "fast", sp.SP_MediaSeekForward)
        return layout

    @classmethod
    def section_header(cls, name):
        return cls._section(name)

    @staticmethod
    def _section(name_):
        def frame():
            f = QFrame()
            f.setFrameStyle(QFrame.HLine | QFrame.Raised)
            return f

        holder = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(frame())
        layout.setContentsMargins(0, 0, 0, 0)
        if name_:
            layout.addWidget(QLabel(name_))
            layout.addWidget(frame())
            for i, s in enumerate([1, 0, 1]):
                layout.setStretch(i, s)
        holder.setLayout(layout)

        return holder

    def __button(self, layout, name, icon, shortcut=None):
        button = QToolButton()
        button.setIcon(self.style().standardIcon(icon))
        if shortcut is not None:
            button.setShortcut(shortcut)
        layout.addWidget(button)
        self.buttons[name] = button
        return button

    def _connect_signals(self):
        reset_maze = functools.partial(self.reset, MainWindow.Reset.MAZE)
        for k in ["width", "height", "seed", "vision", "p_lure", "p_trap"]:
            self.config[k].valueChanged.connect(reset_maze)
        for k in ["traps", "lures", "clues"]:
            c = self.config[k]
            c.dataChanged.connect(reset_maze)
            self.config["with_" + k].clicked.connect(reset_maze)
            self.config["with_" + k].clicked.connect(
                lambda b, c_=c: c_.hide(not b))
        self.config["start"].currentTextChanged.connect(reset_maze)
        self.config["unicursive"].clicked.connect(reset_maze)

        reset_robot = functools.partial(
            self.reset, MainWindow.Reset.ROBOT | MainWindow.Reset.CONTROL)
        for k in ["inputs", "outputs"]:
            self.config[k].currentTextChanged.connect(reset_robot)

        self.config["control"].currentTextChanged.connect(
            lambda: self._generate_controller(
                new_value=self.config['control'].currentText(),
                open_dialog=False))

        for k in ["solution", "robot", "dark", "colorblind"]:
            self.config["show_" + k].clicked.connect(
                lambda v, k_=k: self.maze_w.update_config(**{k_: v}))

        def connect(name, f): self.buttons[name].clicked.connect(f)
        connect("stop", self.stop)
        connect("play", lambda: self._play(None))
        connect("next", self.next)

        save: QAbstractButton = self.buttons["save"]
        save.clicked.connect(self.save)

        connect("c_load",
                lambda: self._generate_controller("",
                                                  open_dialog=True))

    # =========================================================================
    # == Persistent storage
    # =========================================================================

    @staticmethod
    def _settings():
        return QSettings("kgd", "amaze")

    def _restore_settings(self, args):
        config = self._settings()
        logger.info(f"Loading configuration from {config.fileName()}")

        def _try(name, f_=None, default=None):
            if (v_ := config.value(name.replace("_", "/"))) is not None:
                return f_(v_) if f_ else v_
            else:
                return default
        _try("pos", lambda p: self.move(p))
        _try("size", lambda s: self.resize(s))

        def restore_bool(s, k__):
            b_ = bool(int(s))
            self.config[k__].setChecked(b_)
            return b_

        viewer_options = {}
        for k in MazeWidget.default_config().keys():
            k_ = "show_" + k
            b = _try(k_, functools.partial(restore_bool, k__=k_))
            viewer_options[k] = b

        self._restore_from_robot_build_data(
            Robot.BuildData(**_try("robot", default={}))
            .override_with(
                Robot.BuildData.from_argparse(args, set_defaults=False)))

        overrides = Maze.BuildData.from_argparse(args, set_defaults=False)
        if args.maze is not None:
            overrides = Maze.BuildData.from_string(args.maze, overrides)
        self._restore_from_maze_build_data(
            Maze.BuildData(**_try("maze", default={}))
            .override_with(overrides))

        config.beginGroup("sections")
        for k, v in self.sections.items():
            self.sections[k].set_collapsed(
                bool(int(config.value(k.value.lower(), False))))
        config.endGroup()

        return viewer_options

    def _save_settings(self):
        if not self.save_on_exit:
            return

        config = self._settings()
        logger.info(f"Saving configuration to {config.fileName()}")

        config.setValue('pos', self.pos())
        config.setValue('size', self.size())

        config.setValue('maze', self.maze_data().__dict__)
        config.setValue('robot', self._robot_data().__dict__)

        config.beginGroup("show")
        for k in MazeWidget.default_config().keys():
            if (w := self.config.get("show_" + k)) is not None:
                config.setValue(k, int(w.isChecked()))
        config.endGroup()

        config.beginGroup("sections")
        for k, v in self.sections.items():
            config.setValue(k.value.lower(), int(v.collapsed()))
        config.endGroup()

        config.sync()

    def closeEvent(self, _):
        self._save_settings()

    # =========================================================================
    # == Argv/Storage parsing
    # =========================================================================

    def _restore_from_maze_build_data(self, data: Maze.BuildData):
        def val(k): return getattr(data, k)

        def _set(f, *args, **kwargs):
            assert isinstance(f.__self__, QObject)
            _ = QSignalBlocker(f.__self__)
            f(*args, **kwargs)

        for name in ["width", "height", "seed"]:
            if value := val(name):
                _set(self.config[name].setValue, value)

        for name in ["p_lure", "p_trap"]:
            if value := val(name):
                _set(self.config[name].setValue, value * 100)

        for name in ["clue", "lure", "trap"]:
            _set(self.config[name + "s"].set_signs, val(name))

        for name, val_ in [("with_clues", val("clue")),
                           ("with_lures", val("lure")),
                           ("with_traps", val("trap")),
                           ("unicursive", val("unicursive")),
                           ("rotated", val("rotated"))]:
            _set(self.config[name].setChecked, bool(val_))

        for name in ["clues", "lures", "traps"]:
            self.config[name].hide(not self.config["with_" + name].isChecked())

        _set(self.config["start"].setCurrentText, val("start").name.lower())

    def _restore_from_robot_build_data(self, data: Robot.BuildData):
        def val(k): return getattr(data, k)

        for name in ["vision"]:
            self.config[name].setValue(val(name))

        for name in ["inputs", "outputs"]:
            self.config[name].setCurrentText(val(name).name.lower())

        name = "control"
        self.config[name].setCurrentText(val(name).lower())


class _MovieRecorder:
    def __init__(self, viewer, path: Path):
        self.save_intermediates = True
        self.viewer = viewer
        self.path = path
        self.folder = self.path.with_suffix('')
        self.folder.mkdir(parents=True, exist_ok=True)
        self.frames = []

        max_timestep = self.viewer.simulation.deadline
        self.step_format = (f"{{:0{math.ceil(math.log10(max_timestep))}d}}"
                            f".png")

    def _path(self, name=None):
        if name:
            name = name + "_"
        else:
            name = ""
        name += self.step_format.format(self.viewer.simulation.timestep)
        return str(self.folder.joinpath(name))

    def step(self):
        size = 256

        m_img = self.viewer.maze_w.pretty_render(width=size)
        m_size = m_img.width()

        # robot = self.viewer.sections[MainWindow.Sections.ROBOT]
        # img = QImage(robot.rect().size(), QImage.Format_ARGB32)
        # img.fill(Qt.transparent)
        # robot.render(img)
        # img.save(str(self.folder.joinpath(f"robot_{step}.png")))

        vision = self.viewer.visu["img_inputs"]
        v_size = min(vision.width(), vision.height())
        v_ratio = m_size / v_size
        v_offset = (
            (vision.width() - v_size) // 2,
            (vision.height() - v_size) // 2
        )

        v_img = QImage(m_img.size(), m_img.format())
        painter = QPainter(v_img)
        painter.scale(v_ratio, v_ratio)
        vision.render(painter, QPoint(0, 0),
                      QRegion(*v_offset, v_size, v_size))
        painter.end()

        img = QImage(2*size, size, m_img.format())
        painter = QPainter(img)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(0, 0, m_img)
        painter.drawImage(m_size, 0, v_img)
        painter.end()

        path = self._path()

        m_img.save(self._path("maze"))
        v_img.save(self._path("vision"))
        img.save(path)

        if HAS_PIL:
            self.frames.append(PIL.Image.open(path))

    def save(self):
        if HAS_PIL:
            print("Generating", self.path, "...")
            duration = 1000 // len(self.frames)
            self.frames[0].save(self.path, format="GIF",
                                append_images=self.frames,
                                save_all=True, duration=duration,
                                loop=0)
            print("Done!")
        else:
            print("PIL module not installed. Generate gif manually using"
                  " files under", self.folder)
