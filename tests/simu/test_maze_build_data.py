import argparse
import dataclasses

import pytest

from amaze import Maze, StartLocation, Sign

SL = StartLocation

TYPE_ERROR_DATA = [
    dict(width="nan"),
    dict(height="nan"),
    dict(seed="nan"),
    dict(start="down"),
    dict(start=4),
    dict(unicursive="maybe"),
    dict(p_lure="maybe"),
    dict(p_trap="maybe"),
    *[{k: v} for k in ["clue", "lure", "trap"] for v in ["nan", 1.0, Sign(value=1)]],
]
VALUE_ERROR_DATA = [
    dict(width=0),
    dict(width=1000),
    dict(height=0),
    dict(height=1000),
    dict(p_lure=-1.0),
    dict(p_lure=1000.0),
    dict(p_trap=-1.0),
    dict(p_trap=1000.0),
]

ERROR_STRING_DATA = ["not a maze string"]


def _assert_valid_build_data(bd: Maze.BuildData):
    assert isinstance(bd.width, int)
    assert 2 <= bd.width <= 100
    assert isinstance(bd.height, int)
    assert 2 <= bd.height <= 100
    assert isinstance(bd.seed, int)
    assert sum([bd.start is sl for sl in SL]) == 1
    assert isinstance(bd.rotated, bool)
    assert isinstance(bd.unicursive, bool)
    assert bd.p_lure is None or (isinstance(bd.p_lure, float) and 0 <= bd.p_lure <= 1)
    assert bd.p_trap is None or (isinstance(bd.p_trap, float) and 0 <= bd.p_trap <= 1)
    for signs in [bd.clue, bd.lure, bd.trap]:
        for sign in signs:
            assert isinstance(sign.name, str)
            assert isinstance(sign.value, float)
            assert 0 < sign.value <= 1


def _test_build_data(bd: Maze.BuildData):
    print(bd)
    print(bd.to_string())
    _assert_valid_build_data(bd)


def test_maze_build_data(mbd_kwargs):
    bd = Maze.BuildData(**mbd_kwargs)
    _test_build_data(bd)

    permutations = bd.all_rotations()
    assert sum([bd.start == bd_.start for bd_ in permutations]) == 1
    assert sum([bd.start != bd_.start for bd_ in permutations]) == 3
    for bd_ in permutations:
        _test_build_data(bd_)

    for field in dataclasses.fields(bd):
        kwargs = {field.name: Maze.BuildData.Unset()}
        override = bd.where(**kwargs)
        print(override)
        _test_build_data(bd.override_with(override))


def test_maze_build_data_from_string(mbd_kwargs):
    bd = Maze.BuildData(**mbd_kwargs)
    bd_str = bd.to_string()
    bd2 = Maze.BuildData.from_string(bd_str)
    assert bd_str == bd2.to_string()
    assert bd == bd2
    _test_build_data(bd2)

    for field in dataclasses.fields(bd):
        kwargs = {field.name: Maze.BuildData.Unset()}
        override = bd.where(**kwargs)
        print(override)
        _test_build_data(Maze.BuildData.from_string(bd_str, override))


@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        pytest.param(d, e, id="_".join(f"{k}-{v}" for k, v in d.items()))
        for items, e in [
            (TYPE_ERROR_DATA, TypeError),
            (VALUE_ERROR_DATA, ValueError),
        ]
        for d in items
    ],
)
def test_error_maze_build_data(kwargs, error_type):
    with pytest.raises(error_type):
        Maze.BuildData(**kwargs)


@pytest.mark.parametrize("s", ERROR_STRING_DATA)
def test_error_maze_build_data_from_string(s):
    with pytest.raises(ValueError):
        Maze.BuildData.from_string(s)


def test_maze_base_build_data(mbd_kwargs):
    maze = Maze.BuildData(**mbd_kwargs)
    default = Maze.BuildData.from_argparse({}, set_defaults=True)
    unset = Maze.BuildData.from_argparse({}, set_defaults=False)
    print(default)
    print(unset)
    print(maze.override_with(maze))
    print(maze.override_with(unset))
    print(maze.where(width=15))
    print([bool(f) for f in dataclasses.asdict(unset).values()])

    parser = argparse.ArgumentParser(description="Maze.BuildData tester")
    Maze.BuildData.populate_argparser(parser)
    parser.print_help()
    #
    # args = parser.parse_args(["--maze-width", i.name,
    #                           "--robot-outputs", o.name]
    #                          + ([f"--robot-vision={v}"] if v else []))
    # print(args)
    # assert args.maze_inputs == i
    # assert args.maze_outputs == o
    # assert args.maze_vision == v
    # print(robot.override_with(Maze.BuildData.from_argparse(args)))
