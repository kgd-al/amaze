import sys
from typing import Union

import pytest
from PyQt5.QtCore import QEvent, Qt, QTimer, QPoint
from PyQt5.QtGui import QKeyEvent, QHelpEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QLabel,
)

from amaze import (
    qt_application,
    Maze,
    MazeWidget,
    Robot,
    Simulation,
    Sign,
    InputType,
    OutputType,
)
from amaze.bin.main import Options
from amaze.misc.utils import QtOffscreen, qt_offscreen
from amaze.simu.controllers import (
    CheaterController,
    RandomController,
)
from amaze.simu.pos import Pos, Vec
from amaze.simu.types import State
from amaze.visu import MainWindow
from amaze.visu.widgets.combobox import ZoomingComboBox
from amaze.visu.widgets.labels import (
    InputsLabel,
    OutputsLabel,
    ValuesLabel,
    ElidedLabel,
)
from amaze.visu.widgets.lists import SignList

QT_ACTIONS = [Qt.Key_Right, Qt.Key_Up, Qt.Key_Left, Qt.Key_Down]


class TimedOffscreenApplication:
    def __init__(self, capfd, duration=1):
        self.app, self.timer = None, None
        self.duration = round(1000 * duration)
        self.capfd = capfd
        self.error = False

    def __enter__(self):
        self.app = qt_application(start_offscreen=True)
        QTimer.singleShot(self.duration, self.__failure)
        return self.app

    def __exit__(self, type, value, traceback):
        qt_offscreen(False)
        if self.error:
            raise self.TimeOutException(
                f"Application did not finish in {self.duration / 1000:g} seconds"
            )

        oe = self.capfd.readouterr()
        err = oe.err
        err = "\n".join(
            f">  {e}"
            for e in err.split("\n")
            if not e.startswith("This plugin does not support") and e
        )
        if len(err) > 0:
            print(oe.out)
            raise AssertionError(f"Error output is not empty:\n{err}")

        del self.app

    class TimeOutException(Exception):
        pass

    def __failure(self, *_):
        self.app.quit()
        self.error = True


def _post_key_press(app, obj, key):
    app.postEvent(obj, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))


def test_timed_offscreen(capfd):
    def _test():
        with TimedOffscreenApplication(capfd, 1) as app:
            app.exec()

    pytest.raises(TimedOffscreenApplication.TimeOutException, _test)

    def _test():
        with TimedOffscreenApplication(capfd, 1) as app:
            print("Error message", file=sys.stderr)
            QTimer.singleShot(100, app.quit)
            app.exec()

    pytest.raises(AssertionError, _test)


def test_interface_create():
    with QtOffscreen():
        w = MainWindow(runnable=False)
        w.stop()


def test_interface_default():
    with QtOffscreen():
        w = MainWindow()
        w.show()
        w.close()


def test_interface_step():
    with QtOffscreen():
        args = Options()
        args.controller = "cheater"
        args.restore_config = False
        w = MainWindow(args)
        w.next()
        w.start()
        w.pause()
        w.stop()
        w.resize(250, 250)

        args.restore_config = True
        args.width = 750
        MainWindow(args)


def test_interface_robot_mode(capfd, tmp_path):
    with TimedOffscreenApplication(capfd, 2) as app:
        args = Options()
        args.is_robot = True
        args.autoquit = True
        args.dt = 0
        args.robot = "D"
        w = MainWindow(args)
        w.set_maze(Maze.BuildData.from_string("M4_4x4_U"))
        assert w.simulation.robot.pos.aligned() == (0, 0)
        for action in [0, 1, 0, 3, 0, 1, 1, 1]:
            _post_key_press(app, w, QT_ACTIONS[action])
        assert 0 == app.exec()


@pytest.mark.parametrize("maze", ["M4_4x4_U", "M0_C1_t.5_T.5"])
def test_interface_run(maze, capfd, tmp_path):
    with TimedOffscreenApplication(capfd, 5) as app:
        args = Options()
        args.autoquit = True
        args.dt = 0
        args.robot = "D"
        args.controller = "cheater"
        args.maze = maze
        args.movie = tmp_path.joinpath("movie.gif")

        w = MainWindow(args)
        w.show()
        w.start()

        assert 0 == app.exec()

        w.save()


def test_interface_buttons(capfd):
    with TimedOffscreenApplication(capfd, 5) as app:
        w = MainWindow()
        w.show()

        def _cb_next(cb: QComboBox):
            cb.setCurrentIndex((cb.currentIndex() + 1) % cb.count())

        def _sb_next(sb: Union[QSpinBox, QDoubleSpinBox]):
            value = sb.value() + 0.1 * (sb.maximum() - sb.minimum())
            if isinstance(sb, QSpinBox):
                value = round(value)
            sb.setValue(value % sb.maximum() + sb.minimum())

        def _gb_next(gb: QGroupBox):
            QTest.mouseClick(gb, Qt.LeftButton, Qt.NoModifier, QPoint(10, 10))

        actions = {
            QCheckBox: "click",
            QComboBox: _cb_next,
            QGroupBox: _gb_next,
            QSpinBox: _sb_next,
            QDoubleSpinBox: _sb_next,
            SignList: "add_row",
        }

        print("== Config ==")
        for key, widget in w.config.items():
            print(key, widget)
            action = actions[type(widget)]
            if isinstance(action, str):
                getattr(widget, action)()
            else:
                action(widget)

        print("== Buttons ==")
        for key, widget in w.buttons.items():
            print(key, widget)
            if key == "c_load":
                QTimer.singleShot(
                    100,
                    lambda: QTest.keyEvent(
                        QTest.Click, app.focusWidget(), Qt.Key_Escape
                    ),
                )
            widget.click()

        print("== Sections ==")
        for key, widget in w.sections.items():
            print(key, widget)
            QTest.mouseClick(widget.toggle_button, Qt.LeftButton)

        QTimer.singleShot(500, w.close)

        assert 0 == app.exec()


def test_interface_render(capfd, tmp_path):
    with TimedOffscreenApplication(capfd):
        args = Options()
        args.maze = Maze.BuildData(width=10, height=10, seed=10).to_string()

        w = MainWindow(args)
        sol = w.config["show_solution"]
        sol.setChecked(True)
        w.maze_w.render_to_file(tmp_path.joinpath("maze_sol.png"))
        w.config["show_solution"].click()
        w.maze_w.render_to_file(tmp_path.joinpath("maze_no_sol.png"), width=100)


@pytest.mark.parametrize("maze_t", [str, Maze.BuildData, Maze])
@pytest.mark.parametrize("robot", ["D", "H"])
def test_maze_widget(maze_t, robot, capfd, tmp_path):
    with TimedOffscreenApplication(capfd):
        robot = Robot(Robot.BuildData.from_string(robot))

        bd = Maze.BuildData(width=7, height=7, seed=7, unicursive=True)
        if maze_t == Maze.BuildData:
            maze = bd
        elif maze_t == Maze:
            maze = Maze.generate(bd)
        else:
            maze = bd.to_string()

        mw = MazeWidget(maze=maze, robot=robot)
        pytest.raises(ValueError, mw.set_maze, None)
        mw.set_maze(maze)
        print(mw.minimumSize())

        robot.reset(Pos(0.5, 0.5))
        mw.render_to_file(tmp_path.joinpath("maze.png"))
        mw.update_config(robot=False)
        mw.pretty_render().save(str(tmp_path.joinpath("maze_no_robot.png")))


@pytest.mark.parametrize(
    "maze", ["M4_4x4_U", "M10_10x10_U", "M10_10x10", "M10_5x10", "M10_10x5"]
)
def test_maze_widget_plot(maze, capfd, tmp_path):
    with TimedOffscreenApplication(capfd, 10):
        maze = Maze.generate(Maze.BuildData.from_string(maze))
        robot = Robot.BuildData.from_string("D")

        MazeWidget.static_render_to_file(maze, tmp_path.joinpath("maze.png"))

        simulation = Simulation(maze=maze, robot=robot, save_trajectory=True)
        for c_type, arg in [
            (CheaterController, simulation),
            (RandomController, 0),
            (RandomController, 1),
        ]:
            controller = c_type(robot, arg)

            # 0: 70
            # 1: 98

            print("Running with", c_type)
            simulation.reset()
            simulation.run(controller)

            for name, args in [
                ("base", dict()),
                ("square", dict(square=True)),
                ("lside", dict(side=-1)),
                ("rside", dict(side=+1)),
                ("verbose", dict(verbose=2)),
                ("traj", dict(trajectory=simulation.trajectory)),
            ]:
                MazeWidget.plot_trajectory(
                    simulation,
                    256,
                    path=tmp_path.joinpath(f"trajectory_{c_type.name}_{name}.png"),
                    **args,
                )


@pytest.mark.parametrize("controls", [True, False])
def test_sign_list(controls, capfd):
    with TimedOffscreenApplication(capfd) as app:
        sl = SignList(dict(foo=QLabel("bar")) if controls else None)

        n = sl.count()
        assert n == 0
        sl.button_add.click()
        assert sl.count() == n + 1

        signs = [Sign(value=1), Sign(name="point", value=0.5)]
        sl.set_signs(signs)
        assert signs == sl.signs()

        sl.children()[-1].children()[-1].click()
        assert n == 0

        sl.show()
        assert sl.isVisible()
        sl.hide()
        assert not sl.isVisible()
        sl.hide(False)
        assert sl.isVisible()

        sl.show()

        QTimer.singleShot(500, sl.close)
        assert 0 == app.exec()


def test_inputs_label(capfd):
    with TimedOffscreenApplication(capfd, 5) as app:
        il = InputsLabel()
        il.show()
        print("InputsLabel at create, pixmap =", il.pixmap())
        il.update()

        def _set():
            il.set_inputs([0 for _ in range(8)], InputType.DISCRETE)
            print("InputsLabel with inputs, pixmap =", il.pixmap())

        QTimer.singleShot(200, _set)

        QTimer.singleShot(500, il.close)
        assert 0 == app.exec()


def test_outputs_label(capfd):
    with TimedOffscreenApplication(capfd, 5) as app:
        ol = OutputsLabel()
        ol.show()

        def _set():
            ol.set_outputs(Vec(1, 0), OutputType.DISCRETE)

        QTimer.singleShot(200, _set)

        QTimer.singleShot(500, ol.close)
        assert 0 == app.exec()


@pytest.mark.parametrize("values", [[0, 0, 0, 0], [1, 0, 0, 0]])
def test_values_label(values, capfd):
    with TimedOffscreenApplication(capfd, 5) as app:
        vl = ValuesLabel()

        class Decoy:
            def values(self, *_):
                return values

        vl.set_values(Decoy(), State([0]))

        vl.show()
        QTimer.singleShot(500, vl.close)
        assert 0 == app.exec()


@pytest.mark.parametrize("elide", [Qt.ElideNone, Qt.ElideLeft])
def test_elided_label(elide, capfd):
    with TimedOffscreenApplication(capfd, 5) as app:
        el = ElidedLabel("Text", mode=elide)
        el.show()

        def _set():
            el.setText("Another text")

        QTimer.singleShot(300, _set)
        QTimer.singleShot(200, _set)

        QTimer.singleShot(500, el.close)
        assert 0 == app.exec()


def test_zooming_combobox(capfd):
    with TimedOffscreenApplication(capfd, 5) as app:
        cb = ZoomingComboBox(value_getter=lambda: 1)
        cb.addItem("arrow")

        def _show():
            app.postEvent(cb, QHelpEvent(QEvent.ToolTip, QPoint(0, 0), QPoint(0, 0)))

        cb.show()

        QTimer.singleShot(200, _show)
        QTimer.singleShot(500, cb.close)
        assert 0 == app.exec()
