import pprint
from math import floor, ceil

import pytest

from amaze import Maze


def test_error_direct_init():
    with pytest.raises(RuntimeError):
        Maze(Maze.BuildData())


def test_maze_static():
    for action in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        assert Maze.offsets[Maze.offsets_inv[action]] == action


def test_maze(mbd_kwargs, tmpdir):
    bd = Maze.BuildData(**mbd_kwargs)
    maze = Maze.generate(bd)
    assert bd.to_string() == str(maze)

    stats = maze.stats()
    ints = maze.intersections()
    path = len(maze.solution) - ints
    pprint.pprint(mbd_kwargs)
    pprint.pprint(stats)
    if p_lure := bd.p_lure:
        lures = (path - 1) * p_lure
        assert floor(lures) <= stats["lures"] <= ceil(lures)

    if p_trap := bd.p_trap:
        assert floor(ints * p_trap) <= stats["traps"] <= ceil(ints * p_trap)

        p_clue = (1 - p_trap) * (len(maze.clues()) > 0)
        assert floor(ints * p_clue) <= stats["clues"] <= ceil(ints * p_clue)

    else:
        if len(maze.clues()) > 0:
            assert stats["clues"] == ints
        else:
            assert stats["clues"] == 0

    assert sum(1 for _ in maze.iter_cells()) == maze.width * maze.height
    assert sum(1 for _ in maze.solution) == stats["path"]
    assert sum(1 for _ in maze.iter_solutions()) == stats["path"]

    for _bd, _maze in zip(bd.all_rotations(), maze.all_rotations()):
        assert _bd == _maze.build_data()
