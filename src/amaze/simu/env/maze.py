import json
import random
import re
import time
from dataclasses import dataclass, fields, asdict
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Annotated, Tuple, Optional, List

import numpy as np

from amaze.utils.build_data import BaseBuildData
from amaze.visu import resources

logger = getLogger(__name__)


class StartLocation(int, Enum):
    SOUTH_WEST = 0
    SOUTH_EAST = 1
    NORTH_EAST = 2
    NORTH_WEST = 3

    def shorthand(self):
        return ''.join(s[0] for s in self.name.split('_'))

    @classmethod
    def from_shorthand(cls, short):
        return {'SW': cls.SOUTH_WEST, 'SE': cls.SOUTH_EAST,
                'NE': cls.NORTH_EAST, 'NW': cls.NORTH_WEST}[short]


class Maze:
    @dataclass
    class BuildData(BaseBuildData):
        width: Annotated[int, "number of horizontal cells"] = 10
        height: Annotated[int, "number of vertical cells"] = 10

        seed: Annotated[Optional[int], "seed for the RNG"] = None

        start: Annotated[StartLocation, "location of the initial position"] = \
            StartLocation.SOUTH_WEST
        # end: Annotated[Optional[Tuple[int, int]],
        #                "coordinate of the target position"] = None

        unicursive: Annotated[bool, "Single path?"] = False

        cue: Annotated[Optional[str],
                       "image name for (helpful) cues"] = None

        p_trap: Annotated[Optional[float],
                          "probability of generating a trap (per-cell)"] = None

        trap: Annotated[Optional[str],
                        "image name for (unhelpful) traps"] = None

        def __post_init__(self):
            if self.seed is None:
                self.seed = round(time.time() * 1000)

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
        self.e_start = data.start
        self.start = {
            StartLocation.SOUTH_WEST: (0, 0),
            StartLocation.SOUTH_EAST: (self.width-1, 0),
            StartLocation.NORTH_EAST: (self.width-1, self.height-1),
            StartLocation.NORTH_WEST: (0, self.height-1),
        }[self.e_start]
        self.end = (self.width - self.start[0] - 1,
                    self.height - self.start[1] - 1)

        self.seed = data.seed
        self.solution: Optional[List[Tuple[int, int]]] = None
        self.intersections: Optional[List[Tuple[int, Maze.Direction]]] = None

        self.cue: Optional[str] = None
        self.p_trap: float = 0
        self.trap: Optional[str] = None
        self.traps: Optional[List[Tuple[int, Maze.Direction]]] = None

    def __repr__(self):
        return f"{self.width}x{self.height} maze"

    def valid(self, i, j):
        return 0 <= i < self.width and 0 <= j < self.height

    def unicursive(self):
        return len(self.intersections) == 0

    def wall(self, i: int, j: int, d: Direction):
        return self.walls[i, j, d.value]

    def wall_delta(self, i: int, j: int, di: int, dj: int):
        return self.wall(i, j, self._offsets_inv[(di, dj)])

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
        for i, (c_i, c_j) in enumerate(maze.solution[1:-1]):
            if sum(maze.walls[c_i, c_j]) < 2:
                nc_i, nc_j = maze.solution[i + 2]
                d_i, d_j = nc_i - c_i, nc_j - c_j
                intersections.append((i + 1, cls._offsets_inv[(d_i, d_j)]))
        maze.intersections = intersections

        # maze.unicursive = data.unicursive
        if data.unicursive:
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

            # Check last cell
            c_i, c_j = maze.end
            if sum(maze.walls[c_i, c_j]) < 3:
                c0_i, c0_j = maze.solution[-2]
                di, dj = c0_i - c_i, c0_j - c_j
                dirs = cls._offsets_inv.copy()
                dirs.pop((di, dj))
                for _, d_ in dirs.items():
                    maze._set_wall(c_i, c_j, d_, True)

        maze.cue = data.cue
        if maze.cue:
            resources.validate(maze.cue)

        if data.p_trap is not None and data.p_trap > 0 and \
                data.trap is not None:
            maze.p_trap = data.p_trap
            maze.trap = data.trap
            resources.validate(maze.trap)

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

        if not data.unicursive and maze.cue is None:
            logger.warning("Mazes with intersections and no clues are"
                           " practically unsolvable")

        return maze

    def _build_data(self):
        return Maze.BuildData(
            width=self.width, height=self.height,
            seed=self.seed,
            start=self.e_start,
            unicursive=self.unicursive(),
            cue=self.cue,
            p_trap=self.p_trap, trap=self.trap
        )

    def save(self, path: Path):
        bd = self._build_data()
        dct = dict(name=self.bd_to_string(bd))
        dct.update(asdict(bd))
        with open(path, 'w') as f:
            json.dump(dct, f)

    @staticmethod
    def bd_to_string(bd: BuildData):
        default = Maze.BuildData()
        f = f"m{bd.seed}_{bd.width}x{bd.height}"
        if bd.start != default.start:
            f += "_" + bd.start.shorthand()
        if bd.unicursive:
            f += "_U"
        if bd.cue:
            f += f"_C{bd.cue.replace('_', '-')}"
        if bd.trap:
            f += f"_p{bd.p_trap}_T{bd.trap.replace('_', '-')}"
        return f

    def to_string(self):
        return self.bd_to_string(self._build_data())

    @classmethod
    def from_string(cls, s, overrides: Optional[BuildData] = None) -> 'Maze':
        d_re = re.compile("[SN][WE]")
        s_re = re.compile("[0-9]+x[0-9]+")
        bd = cls.BuildData()
        for token in s.split('_'):
            t, tail = token[0], token[1:]
            if t == 'm':
                bd.seed = int(tail)
            elif t == 'U':
                bd.unicursive = True
            elif t == 'C':
                bd.cue = tail.replace('-', '_')
            elif t == 'T':
                bd.trap = tail.replace('-', '_')
            elif t == 'p':
                bd.p_trap = float(tail)
            elif s_re.match(token):
                bd.width, bd.height = [int(x) for x in token.split('x')]
            elif d_re.match(token):
                bd.start = StartLocation.from_shorthand(token)
            else:
                raise ValueError(f"Unknown or malformed token '{token}'")

        if overrides:
            for field in fields(bd):
                if not isinstance(v := getattr(overrides, field.name),
                                  BaseBuildData.Unset):
                    setattr(bd, field.name, v)

        return cls.generate(bd)
