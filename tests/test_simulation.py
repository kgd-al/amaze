import pprint
from typing import Type, Optional, Union

import pytest

from amaze import Maze, Robot, Simulation
from amaze.simu.controllers.cheater import CheaterController
from amaze.simu.controllers.random import RandomController
from amaze.simu.simulation import MazeMetrics

param = pytest.param


def _run_simulation(
    maze: Maze.BuildData,
    robot: str,
    controller_type: Type[Union[RandomController, CheaterController]],
    controller_kwargs: Optional[dict] = None,
    **kwargs,
):
    maze = Maze.generate(maze)
    robot = Robot.BuildData.from_string(robot)
    simulation = Simulation(maze, robot, **kwargs)
    kwargs = controller_kwargs or {}
    if getattr(controller_type, "cheats", False):
        kwargs["simulation"] = simulation
    controller = controller_type(robot_data=robot, **kwargs)
    simulation.run(controller)

    assert simulation.done()
    assert simulation.success() or simulation.failure()

    return simulation, controller


def test_simulation_static():
    print(Simulation.discrete_actions())


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize("robot", "DHC")
def test_simulation(maze_str, robot, save):
    simulation, controller = _run_simulation(
        Maze.BuildData.from_string(maze_str).where(width=5, height=5),
        robot,
        CheaterController,
        save_trajectory=save,
    )
    stats, infos = simulation.stats, simulation.infos()
    time, reward = simulation.time(), simulation.cumulative_reward()

    maze = simulation.maze
    metrics = Simulation.compute_metrics(maze, simulation.data.inputs, simulation.data.vision)
    pprint.pprint(metrics)
    assert 0 < metrics["n_inputs"]["all"] <= maze.width * maze.height * 3
    assert 0 < metrics["n_inputs"]["path"] <= len(maze.solution)
    assert all(0 <= v for v in metrics[MazeMetrics.SURPRISINGNESS].values())
    assert 0 <= metrics[MazeMetrics.DECEPTIVENESS]

    for reset_kwargs in [dict(save_trajectory=save), dict()]:
        simulation.reset(**reset_kwargs)
        controller.reset()
        simulation.run(controller)
        assert stats == simulation.stats
        assert infos == simulation.infos()
        assert time == simulation.time()
        assert reward == simulation.cumulative_reward()


@pytest.mark.slow
@pytest.mark.parametrize("robot", "DHC")
def test_simulation_success(mbd_kwargs, robot):
    simulation, _ = _run_simulation(
        Maze.BuildData(**mbd_kwargs),
        robot,
        CheaterController,
        save_trajectory=False,
    )
    assert simulation.success()
    assert not simulation.failure()


@pytest.mark.parametrize("robot", "DHC")
def test_simulation_success_fast(maze_str, robot):
    simulation, _ = _run_simulation(
        Maze.BuildData.from_string(maze_str),
        robot,
        CheaterController,
        save_trajectory=False,
    )
    assert simulation.success()
    assert not simulation.failure()


def test_simulation_many(mbd_kwargs):
    simulation, _ = _run_simulation(
        Maze.BuildData(**mbd_kwargs),
        "D",
        RandomController,
        controller_kwargs=dict(seed=0),
        save_trajectory=True,
    )


@pytest.mark.parametrize("robot", ["D", "H", "H17"])
def test_inputs_evaluation(maze_str, robot, tmp_path):
    maze = Maze.from_string(maze_str)
    robot = Robot.BuildData.from_string(robot)
    controller = RandomController(seed=0, robot_data=robot)
    simulation = Simulation(maze, robot)

    path = tmp_path.joinpath("member")
    path.mkdir(exist_ok=True, parents=True)
    Simulation.inputs_evaluation_from(
        simulation=simulation,
        results_path=path,
        controller=controller,
        draw_inputs=False,
        draw_individual_files=False,
        draw_summary_file=True,
        summary_file_ratio=1,
    )

    path = tmp_path.joinpath("static")
    path.mkdir(exist_ok=True, parents=True)
    Simulation.inputs_evaluation(
        results_path=str(path),
        controller=controller,
        signs=maze.signs,
        draw_inputs=True,
        draw_individual_files=True,
        draw_summary_file=False,
    )

    path = tmp_path.joinpath("csv")
    path.mkdir(exist_ok=True, parents=True)
    Simulation.inputs_evaluation(
        results_path=path,
        controller=controller,
        signs=maze.signs,
        draw_inputs=False,
        draw_individual_files=False,
        draw_summary_file=False,
    )


def test_error_inputs_evaluation(tmp_path):
    with pytest.raises(ValueError):
        robot = Robot.BuildData.from_string("C")
        controller = RandomController(seed=0, robot_data=robot)
        Simulation.inputs_evaluation(tmp_path, controller, Maze.generate(Maze.BuildData()))
