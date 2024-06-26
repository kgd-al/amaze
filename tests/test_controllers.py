import itertools
from typing import Type, Optional

import pytest

from amaze import Maze, Robot, Simulation
from amaze.simu.controllers.base import BaseController
from amaze.simu.controllers.cheater import CheaterController
from amaze.simu.controllers.keyboard import KeyboardController
from amaze.simu.controllers.random import RandomController
from amaze.simu.controllers.tabular import TabularController

CONTROLLERS = [
    *[pytest.param(t, d, r, id=f"{t_id}-{r}")
      for (t, d, t_id), r in itertools.product(
            [(CheaterController, dict(), "cheater"),
             (RandomController, dict(seed=0), "random")],
            "DHC")
      ],
    pytest.param(TabularController,
                 dict(actions=TabularController.discrete_actions,
                      epsilon=0.1, seed=0),
                 "D", id="tabular")
]


def _run_simulation(maze: Maze.BuildData, robot: str,
                    controller_type: Type[BaseController],
                    controller_kwargs: Optional[dict] = None,
                    **kwargs: dict):
    maze = Maze.generate(maze)
    robot = Robot.BuildData.from_string(robot)
    simulation = Simulation(maze, robot, **kwargs)
    kwargs = controller_kwargs or {}
    if controller_type.cheats:
        kwargs["simulation"] = simulation
    controller = controller_type(robot_data=robot, **kwargs)
    simulation.run(controller)

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
    pytest.raises(NotImplementedError, controller.load_from_archive, None)


def test_controller_cheater():
    pytest.raises(ValueError, CheaterController,
                  Robot.BuildData.from_string("H"), simulation=None)
