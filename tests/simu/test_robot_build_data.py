import argparse
import dataclasses

import pytest

from amaze import Robot, InputType, OutputType, Maze, Simulation
from amaze.simu.controllers.cheater import CheaterController

IT = InputType
OT = OutputType


DATA = [
    pytest.param(IT.DISCRETE, OT.DISCRETE, None, id="discrete"),
    pytest.param(IT.DISCRETE, OT.DISCRETE, 11, id="discrete_vision"),
    pytest.param(IT.CONTINUOUS, OT.DISCRETE, None, id="hybrid"),
    pytest.param(IT.CONTINUOUS, OT.DISCRETE, 11, id="hybrid_vision"),
    pytest.param(IT.CONTINUOUS, OT.CONTINUOUS, None, id="continuous"),
    pytest.param(IT.CONTINUOUS, OT.CONTINUOUS, 11, id="continuous_vision"),
]


ERROR_DATA = [
    pytest.param("D", OT.DISCRETE, None, id="istr"),
    pytest.param(IT.DISCRETE, "D", None, id="ostr"),
    pytest.param(IT.DISCRETE, OT.DISCRETE, "v", id="vstr"),
    pytest.param(OT.DISCRETE, IT.DISCRETE, 11, id="swap"),
]


STRING_DATA = [
    "DD",
    "DD11",
    "CD",
    "CD11",
    "CC",
    "CC11",
    "D",
    "D11",
    "H",
    "H11",
    "D",
    "D11",
]


TYPE_ERROR_STRING_DATA = ["A", "AD", "DA", "AA", "A11", "AAA", "CDA"]

VALUE_ERROR_STRING_DATA = ["DC", "H10"]


def _assert_valid_build_data(bd: Robot.BuildData):
    assert isinstance(bd.inputs, InputType)
    assert isinstance(bd.outputs, OutputType)
    assert bd.vision is None or isinstance(bd.vision, int)
    assert (bd.inputs is IT.CONTINUOUS) == (bd.vision is not None)


def _test_build_data(bd: Robot.BuildData):
    print(bd)
    print(bd.to_string())
    _assert_valid_build_data(bd)


@pytest.mark.parametrize("i,o,v", DATA)
def test_robot_build_data(i: InputType, o: OutputType, v: int):
    bd = Robot.BuildData(i, o, v)
    assert bd.inputs == i
    assert bd.outputs == o
    _test_build_data(bd)

    bd_str = bd.to_string()
    bd2 = Robot.BuildData.from_string(bd_str)
    assert bd_str == bd2.to_string()
    assert bd == bd2


@pytest.mark.parametrize("s", STRING_DATA)
def test_robot_build_data_from_string(s):
    assert Robot.BuildData.from_string(s) == Robot.BuildData.from_string(s)

    bd = Robot.BuildData.from_string(s)
    _test_build_data(bd)

    override = Robot.BuildData(
        (InputType.DISCRETE if bd.inputs is InputType.CONTINUOUS else InputType.CONTINUOUS),
        (OutputType.DISCRETE if bd.outputs is OutputType.CONTINUOUS else OutputType.CONTINUOUS),
        None if bd.vision is not None else 11,
    )
    assert Robot.BuildData.from_string(s, override) == override


@pytest.mark.parametrize("i,o,v", ERROR_DATA)
def test_error_robot_build_data(i: InputType, o: OutputType, v: int):
    with pytest.raises(TypeError):
        Robot.BuildData(i, o, v)


@pytest.mark.parametrize(
    "s,t",
    [
        *[(s, TypeError) for s in TYPE_ERROR_STRING_DATA],
        *[(s, ValueError) for s in VALUE_ERROR_STRING_DATA],
    ],
)
def test_error_robot_build_data_from_string(s, t):
    with pytest.raises(t):
        Robot.BuildData.from_string(s)


@pytest.mark.parametrize("i,o,v", DATA)
def test_robot_build_data_from_controller(i: InputType, o: OutputType, v: int):
    robot = Robot.BuildData(i, o, v)
    maze = Maze.from_string("M4_10x10_U")
    simulation = Simulation(maze, robot)

    controller = CheaterController(robot, simulation)
    action = controller(simulation.observations)
    simulation.step(action)
    print(simulation.robot.to_dict())

    if robot.outputs is OutputType.CONTINUOUS:
        simulation.step(-1 * action)

    bd = Robot.BuildData.from_controller(controller)
    assert robot.inputs == bd.inputs
    assert robot.outputs == bd.outputs
    assert robot.vision == bd.vision
    _test_build_data(bd)


@pytest.mark.parametrize("i,o,v", DATA)
def test_robot_base_build_data(i: InputType, o: OutputType, v: int):
    robot = Robot.BuildData(i, o, v)
    default = Robot.BuildData.from_argparse({}, set_defaults=True)
    unset = Robot.BuildData.from_argparse({}, set_defaults=False)
    print(default)
    print(unset)
    print(robot.override_with(robot))
    print(robot.override_with(unset))
    print(robot.where(vision=15))
    print([bool(f) for f in dataclasses.asdict(unset).values()])

    parser = argparse.ArgumentParser(description="Robot.BuildData tester")
    Robot.BuildData.populate_argparser(parser)
    parser.print_help()

    args = parser.parse_args(
        ["--robot-inputs", i.name, "--robot-outputs", o.name]
        + ([f"--robot-vision={v}"] if v else [])
    )
    print(args)
    assert args.robot_inputs == i
    assert args.robot_outputs == o
    assert args.robot_vision == v
    print(robot.override_with(Robot.BuildData.from_argparse(args)))
