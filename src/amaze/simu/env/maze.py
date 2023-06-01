import random
import time
from ctypes import c_int32
from dataclasses import dataclass, fields
from enum import Enum
from logging import getLogger
from typing import Annotated, Tuple, Optional, Union, List, get_args, get_origin

import numpy as np

from amaze.visu import resources


logger = getLogger(__name__)


class Maze:
    @dataclass
    class BuildData:
        width: Annotated[int, "number of horizontal cells"] = 10
        height: Annotated[int, "number of vertical cells"] = 10

        seed: Annotated[Optional[int], "seed for the RNG"] = None

        start: Annotated[Tuple[int, int],
                         "coordinate of the initial position"] = (0, 0)
        end: Annotated[Optional[Tuple[int, int]],
                       "coordinate of the target position"] = None

        unicursive: Annotated[bool, "Single path?"] = False

        cue: Annotated[Optional[str],
                       "image name for (helpful) cues"] = None

        p_trap: Annotated[Optional[float],
                          "probability of generating a trap (per-cell)"] = None

        trap: Annotated[Optional[str],
                        "image name for (unhelpful) traps"] = None

        def __post_init__(self):
            if self.seed is None:
                self.seed = c_int32(round(time.time() * 1000)).value
            if self.end is None:
                self.end = (self.width-1, self.height-1)

        @classmethod
        def populate_argparser(cls, parser):
            for field in fields(cls):
                a_type = field.type.__args__[0]
                t_args = get_args(a_type)
                if get_origin(a_type) is Union and type(None) in t_args:
                    f_type = t_args[0]
                else:
                    f_type = a_type

                help_msg = \
                    f"{'.'.join(field.type.__metadata__)}" \
                    f" (default: {field.default}," \
                    f" type: {f_type})"
                parser.add_argument("--maze-" + field.name,
                                    dest="maze_" + field.name,
                                    default=None,
                                    type=f_type, help=help_msg)

        @classmethod
        def from_argparse(cls, namespace):
            print(namespace.__dict__)
            data = cls()
            for field in fields(cls):
                f_name = "maze_" + field.name
                if hasattr(namespace, f_name) \
                        and (attr := getattr(namespace, f_name)) is not None:
                    setattr(data, field.name, attr)
                else:
                    setattr(data, field.name, None)
            return data

    class Direction(Enum):
        EAST = 0
        NORTH = 1
        WEST = 2
        SOUTH = 3

    _offsets = {
        Direction.EAST: (+1, +0),
        Direction.NORTH: (+0, +1),
        Direction.WEST: (-1, +0),
        Direction.SOUTH: (+0, -1)
    }

    _offsets_inv = {v: k for k, v in _offsets.items()}

    _inverse_dir = {
        Direction.EAST: Direction.WEST,
        Direction.NORTH: Direction.SOUTH,
        Direction.WEST: Direction.EAST,
        Direction.SOUTH: Direction.NORTH
    }

    __private_key = object()

    def __init__(self, data: BuildData, key=None):
        if key is not self.__private_key:
            raise AssertionError("Cannot create maze directly")

        self.width, self.height = data.width, data.height
        self.walls = np.ones((data.width, data.height, 4), dtype=int)
        self.start, self.end = data.start, data.end
        self.seed = data.seed
        self.solution: Optional[List[Tuple[int, int]]] = None
        self.intersections: Optional[List[Tuple[int, Maze.Direction]]] = None

        self.cue = None
        self.trap = None
        self.traps: Optional[List[Tuple[int, Maze.Direction]]] = None

    def __repr__(self):
        return f"{self.width}x{self.height} maze"

    def valid(self, i, j):
        return 0 <= i < self.width and 0 <= j < self.height

    def wall(self, i: int, j: int, d: Direction):
        return self.walls[i, j, d.value]

    def wall_delta(self, i: int, j: int, delta: Tuple[int, int]):
        return self.wall(i, j, self._offsets_inv[delta])

    def direction_from_offset(self, i: int, j: int):
        return self._offsets_inv[(i, j)]

    def _set_wall(self, i: int, j: int, d: Direction, wall: bool):
        od = self._offsets[d]
        d_ = self._inverse_dir[d]
        i_, j_ = i + od[0], j + od[1]
        self.walls[i, j, d.value] = wall
        if self.valid(i_, j_):
            self.walls[i_, j_, d_.value] = wall

    @classmethod
    def generate(cls, data: BuildData):
        maze = Maze(data, cls.__private_key)
        rng = random.Random(maze.seed)

        stack = [maze.start]
        visited = np.full((maze.width, maze.height), False)
        visited[maze.start] = True

        walls = []

        while len(stack) > 0:
            if stack[-1] == maze.end:
                maze.solution = stack.copy()

            c_i, c_j = current = stack.pop()
            neighbors = []
            for d, (di, dj) in cls._offsets.items():
                i, j = c_i + di, c_j + dj
                if maze.valid(i, j) and not visited[i, j]:
                    neighbors.append((d, (i, j)))
            if len(neighbors) > 0:
                d_chosen, chosen = rng.choice(neighbors)
                visited[chosen] = True
                walls.append((current, d_chosen))
                stack.extend([current, chosen])

        for (i, j), d in walls:
            maze._set_wall(i, j, d, False)

        intersections = []
        for i, (c_i, c_j) in enumerate(maze.solution[1:-2]):
            if sum(maze.walls[c_i, c_j]) < 2:
                nc_i, nc_j = maze.solution[i + 2]
                d_i, d_j = nc_i - c_i, nc_j - c_j
                intersections.append((i + 1, cls._offsets_inv[(d_i, d_j)]))
        maze.intersections = intersections

        maze.unicursive = data.unicursive
        if maze.unicursive:
            for i, d in maze.intersections:
                c0_i, c0_j = maze.solution[i-1]
                c1_i, c1_j = maze.solution[i]
                di, dj = c0_i - c1_i, c0_j - c1_j
                dirs = cls._offsets_inv.copy()
                dirs.pop((di, dj))
                dirs.pop(cls._offsets[d])
                for _, d_ in dirs.items():
                    maze._set_wall(c1_i, c1_j, d_, True)
            maze.intersections = []

        maze.cue = data.cue

        if data.p_trap is not None and data.p_trap > 0 and \
                data.trap is not None:
            maze.trap = data.trap

            candidates = \
                set(range(len(maze.solution)-1))\
                - {i[0] for i in maze.intersections}
            maze.traps = []
            for i in rng.sample(candidates,
                                round(data.p_trap * len(candidates) / 100)):
                c_i, c_j = maze.solution[i]
                nc_i, nc_j = maze.solution[i+1]
                dirs = cls._offsets_inv.copy()
                dirs.pop((nc_i - c_i, nc_j - c_j))
                maze.traps.append((i, rng.choice(list(dirs.values()))))

        return maze
