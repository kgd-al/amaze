import itertools
from random import Random
from typing import Type, Optional, Callable

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent

from amaze import Maze, Robot, Simulation, qt_application
from amaze.simu.controllers.base import BaseController
from amaze.simu.controllers.cheater import CheaterController
from amaze.simu.controllers.keyboard import KeyboardController
from amaze.simu.controllers.random import RandomController
from amaze.simu.controllers.tabular import TabularController
from amaze.simu.types import State, Action

TABULAR_ARGS = dict(actions=TabularController.discrete_actions,
                    epsilon=0.1, seed=0)

CONTROLLERS = [
    *[pytest.param(t, d, r, id=f"{t_id}-{r}")
      for (t, d, t_id), r in itertools.product(
            [(CheaterController, dict(), "cheater"),
             (RandomController, dict(seed=0), "random")],
            "DHC")
      ],
    pytest.param(TabularController, TABULAR_ARGS, "D", id="tabular")
]


def _run_simulation(maze: Maze.BuildData, robot: str,
                    controller_type: Type[BaseController],
                    controller_kwargs: Optional[dict] = None,
                    pre_step: Callable[[Simulation, BaseController],
                                       None] = None,
                    post_step: Callable[[Simulation, BaseController, float],
                                        None] = None,
                    **kwargs: dict):

    if controller_type is KeyboardController:
        _ = qt_application()

    maze = Maze.generate(maze)
    robot = Robot.BuildData.from_string(robot)
    simulation = Simulation(maze, robot, **kwargs)
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


@pytest.mark.slow
@pytest.mark.parametrize("c_type, c_args, robot", CONTROLLERS)
def test_controllers(mbd_kwargs, robot,
                     c_type: Type[BaseController],
                     c_args: dict):  # pragma: no cover
    _run_simulation(
        Maze.BuildData(**mbd_kwargs),
        robot, c_type, c_args)


@pytest.mark.parametrize("c_type, c_args, robot", CONTROLLERS)
def test_controllers_fast(maze_str, robot,
                          c_type: Type[BaseController], c_args: dict):
    simulation, controller = _run_simulation(
        Maze.BuildData.from_string(maze_str),
        robot, c_type, c_args)

    infos = simulation.infos()

    print(controller.name(), controller.details())
    print(controller.input_type, controller.output_type, controller.vision)
    print(controller.inputs_types(), controller.outputs_types())
    print(controller.is_simple)

    controller.reset()
    simulation.run(controller)
    assert infos == simulation.infos()

    # Corner case of CheaterController
    print(controller(simulation.observations))
    print(controller(simulation.observations))


@pytest.mark.parametrize("c_type, c_args, robot",
                         [param for param in CONTROLLERS
                          if not param.values[0].savable])
def test_controller_not_implemented(c_type, c_args, robot):
    robot = Robot.BuildData.from_string(robot)
    c_args = dict(**c_args)
    if c_type.cheats:
        c_args["simulation"] = Simulation(Maze.from_string(""), robot)
    c_args["robot_data"] = robot
    controller = c_type(**c_args)

    pytest.raises(NotImplementedError, controller.save_to_archive, None)
    pytest.raises(NotImplementedError,
                  controller.load_from_archive, None, None)


def test_controller_cheater():
    pytest.raises(ValueError, CheaterController,
                  Robot.BuildData.from_string("H"), simulation=None)


@pytest.mark.parametrize("learner", ["sarsa", "q_learning"])
def test_controller_tabular(maze_str, learner, tmp_path):
    for robot in "HC":
        pytest.raises(ValueError, TabularController,
                      Robot.BuildData.from_string(robot), **TABULAR_ARGS)

    rng = Random(0)

    def fake_training(s: Simulation, c: BaseController, r: float):
        getattr(c, learner)(s.observations,
                            rng.choice(TabularController.discrete_actions),
                            r, s.observations,
                            rng.choice(TabularController.discrete_actions),
                            .5, .5)

    _, controller = _run_simulation(
        Maze.BuildData.from_string(maze_str), "D",
        TabularController, TABULAR_ARGS, post_step=fake_training)

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
def test_controller_keyboard(maze_str, robot):
    print(maze_str, robot)

    app = qt_application(allow_create=True)
    rng = Random(0)
    keys = [Qt.Key_Down, Qt.Key_Up, Qt.Key_Left, Qt.Key_Right]

    def fake_events(_, c, *args):
        for t in [QKeyEvent.KeyPress, QKeyEvent.KeyRelease]:
            app.postEvent(c, QKeyEvent(t, rng.choice(keys), Qt.NoModifier))
        app.processEvents()

    _, controller = _run_simulation(
        Maze.BuildData.from_string(maze_str), robot,
        KeyboardController, dict(), pre_step=fake_events)

# def test_control
