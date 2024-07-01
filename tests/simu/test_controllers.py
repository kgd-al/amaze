import itertools
from random import Random
from typing import Type, Optional, Callable
from zipfile import ZipFile

import pytest
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QKeyEvent

from amaze import Maze, Robot, Simulation, qt_application, load
from amaze.simu.controllers import (
    controller_factory,
    BaseController,
    CheaterController,
    RandomController,
    KeyboardController,
    TabularController,
    builtin_controllers,
)
from amaze.simu.controllers.control import save
from amaze.simu.pos import Vec
from amaze.simu.types import OutputType

TABULAR_ARGS = dict(actions=TabularController.discrete_actions, epsilon=0.1, seed=0)

CONTROLLERS = [
    *[
        pytest.param(t, d, r, id=f"{t_id}-{r}")
        for (t, d, t_id), r in itertools.product(
            [
                (CheaterController, dict(), "cheater"),
                (RandomController, dict(seed=0), "random"),
            ],
            "DHC",
        )
    ],
    pytest.param(TabularController, TABULAR_ARGS, "D", id="tabular"),
]


def _make_simulation(maze: Maze.BuildData, robot: str, **kwargs):
    maze = Maze.generate(maze)
    robot = Robot.BuildData.from_string(robot)
    simulation = Simulation(maze, robot, **kwargs)
    return maze, robot, simulation


def _run_simulation(
    maze: Maze.BuildData,
    robot: str,
    controller_type: Type[BaseController],
    controller_kwargs: Optional[dict] = None,
    pre_step: Callable[[Simulation, BaseController], None] = None,
    post_step: Callable[[Simulation, BaseController, float], None] = None,
    **kwargs: dict,
):

    maze, robot, simulation = _make_simulation(maze, robot, **kwargs)

    kwargs = controller_kwargs or {}
    if controller_type.cheats:
        kwargs["simulation"] = simulation
    controller = controller_type(robot_data=robot, **kwargs)

    while not simulation.done():
        if pre_step is not None:
            pre_step(simulation, controller)
        reward = simulation.step(controller(simulation.observations))
        if post_step is not None:
            post_step(simulation, controller, reward)

    assert simulation.done()
    assert simulation.success() or simulation.failure()

    return simulation, controller


def _unsavable(controller: BaseController, tmp_path):
    assert not controller.savable
    path = tmp_path.joinpath("bad.zip")
    pytest.raises(NotImplementedError, controller.save, path)
    pytest.raises(NotImplementedError, controller._save_to_archive, None)
    pytest.raises(NotImplementedError, controller.load, path)
    pytest.raises(NotImplementedError, controller._load_from_archive, None, None)
    pytest.raises(NotImplementedError, controller.assert_equal, None, None)


@pytest.mark.slow
@pytest.mark.parametrize("c_type, c_args, robot", CONTROLLERS)
def test_controllers(
    mbd_kwargs, robot, c_type: Type[BaseController], c_args: dict
):  # pragma: no cover
    _run_simulation(Maze.BuildData(**mbd_kwargs), robot, c_type, c_args)


@pytest.mark.parametrize("c_type, c_args, robot", CONTROLLERS)
def test_controllers_fast(maze_str, robot, c_type: Type[BaseController], c_args: dict):
    simulation, controller = _run_simulation(
        Maze.BuildData.from_string(maze_str), robot, c_type, c_args
    )

    infos = simulation.infos()

    print(controller.name, controller.details())
    print(controller.input_type, controller.output_type, controller.vision)
    print(controller.inputs_types(), controller.outputs_types())
    print(controller.is_simple)

    controller.reset()
    simulation.run(controller)
    assert infos == simulation.infos()

    # Corner case of CheaterController
    print(controller(simulation.observations))
    print(controller(simulation.observations))


@pytest.mark.parametrize(
    "c_type, c_args, robot",
    [param for param in CONTROLLERS if not param.values[0].savable],
)
def test_controller_not_implemented(c_type, c_args, robot, tmp_path):
    robot = Robot.BuildData.from_string(robot)
    c_args = dict(**c_args)
    if c_type.cheats:
        c_args["simulation"] = Simulation(Maze.from_string(""), robot)
    c_args["robot_data"] = robot
    controller = c_type(**c_args)

    _unsavable(controller, tmp_path)


def test_controller_cheater():
    pytest.raises(
        ValueError,
        CheaterController,
        Robot.BuildData.from_string("H"),
        simulation=None,
    )


@pytest.mark.parametrize("learner", ["sarsa", "q_learning"])
def test_controller_tabular(maze_str, learner, tmp_path):
    for robot in "HC":
        pytest.raises(
            ValueError,
            TabularController,
            Robot.BuildData.from_string(robot),
            **TABULAR_ARGS,
        )

    rng = Random(0)

    def fake_training(s: Simulation, c: BaseController, r: float):
        getattr(c, learner)(
            s.observations,
            rng.choice(TabularController.discrete_actions),
            r,
            s.observations,
            rng.choice(TabularController.discrete_actions),
            0.5,
            0.5,
        )

    _, controller = _run_simulation(
        Maze.BuildData.from_string(maze_str),
        "D",
        TabularController,
        TABULAR_ARGS,
        post_step=fake_training,
    )

    controller: TabularController
    print(controller.states())
    for updates in [True, False]:
        controller.pretty_print(show_updates=updates)

    path = tmp_path.joinpath("tabular.zip")
    print("Saving to", path)
    controller.save(path)
    print("Loading from", path)
    controller_roundabout = TabularController.load(path)

    TabularController.assert_equal(controller, controller_roundabout)


@pytest.mark.parametrize("robot", "DHC")
def test_controller_keyboard(maze_str, robot, tmp_path):
    print(maze_str, robot)

    app = qt_application(allow_create=True, start_offscreen=True)
    rng = Random(0)
    keys = [Qt.Key_Down, Qt.Key_Up, Qt.Key_Left, Qt.Key_Right]

    actions = []

    def fake_key(c, k):
        for t in [QKeyEvent.KeyPress, QKeyEvent.KeyRelease]:
            app.postEvent(c, QKeyEvent(t, k, Qt.NoModifier))
        app.processEvents()

    def fake_events(_, c, *args):
        key = rng.choice(keys)
        actions.append(key)
        fake_key(c, key)

    simulation, controller = _run_simulation(
        Maze.BuildData.from_string(maze_str),
        robot,
        KeyboardController,
        dict(),
        pre_step=fake_events,
    )

    infos = simulation.infos()

    controller.reset()
    simulation.reset()

    pos = simulation.robot.pos
    original_pos = pos.copy()

    def test_ignored_event(e_type, *args):
        app.postEvent(controller, e_type(*args))
        app.processEvents()
        assert pos == original_pos
        assert not controller.eventFilter(controller, e_type(*args))

    test_ignored_event(QEvent, QEvent.None_)
    test_ignored_event(QKeyEvent, QKeyEvent.KeyPress, Qt.Key_Down, Qt.ControlModifier)
    if controller.output_type is OutputType.DISCRETE:
        test_ignored_event(QKeyEvent, QKeyEvent.KeyRelease, Qt.Key_Up, Qt.NoModifier)
    test_ignored_event(QKeyEvent, QKeyEvent.ShortcutOverride, Qt.Key_Down, Qt.NoModifier)
    test_ignored_event(QKeyEvent, QKeyEvent.KeyPress, Qt.Key_Enter, Qt.NoModifier)

    controller.reset()
    simulation.reset()

    while not simulation.done():
        fake_key(controller, actions[simulation.timestep])
        simulation.step(controller(simulation.observations))

    assert infos == simulation.infos()
    assert controller(simulation.observations) == Vec.null()
    _unsavable(controller, tmp_path)


@pytest.mark.parametrize("c_type, c_args, robot", CONTROLLERS)
def test_control(c_type, c_args, robot, tmp_path):
    print(builtin_controllers())
    print(c_type, c_type.short_name)

    c_args = c_args.copy()
    c_args["simulation"] = _make_simulation(Maze.BuildData(), robot)[-1]
    robot = Robot.BuildData.from_string(robot)
    c_args["robot_data"] = robot

    controller = controller_factory(c_type.short_name, c_args)

    if c_type.savable:
        base_path = tmp_path.joinpath(c_type.short_name)

        def roundabout(fn, *args):
            _path = save(controller, fn(base_path), *args)
            controller_roundabout = load(_path)
            controller.assert_equal(controller, controller_roundabout)

        roundabout(lambda p: p)
        roundabout(lambda p: p.with_suffix(".zip"))
        roundabout(lambda p: str(p) + "_str")
        roundabout(
            lambda p: p.with_name(c_type.short_name + "_infos"),
            dict(name=c_type.short_name),
        )


def test_error_control(tmp_path):
    pytest.raises(ValueError, save, True, "foo")

    for t in ["bool", "extension.bool"]:
        path = tmp_path.joinpath(f"bad-${t}.zip")
        with ZipFile(path, "w") as archive:
            archive.writestr("controller_class", t)

        pytest.raises(ValueError, load, path)
