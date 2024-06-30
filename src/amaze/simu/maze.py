"""Data structures describing a maze and its parameters
"""

import random
import re
import time
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice, cycle
from logging import getLogger
from types import SimpleNamespace
from typing import Annotated, Tuple, Optional, List, Dict

import numpy as np

from amaze.misc.resources import Sign, SignType
from ._build_data import BaseBuildData
from .types import StartLocation, classproperty

logger = getLogger(__name__)


class Maze:
    """Main data structure storing everything needed during simulation"""

    @dataclass
    class BuildData(BaseBuildData):
        """Structure describing all of a mazes parameters.

        Used during building and thrown afterward. Also used for
        string conversion.
        """

        _FIELD_SEP = "_"

        width: Annotated[int, "number of horizontal cells"] = 10
        """ The width of the maze
        """

        #: The height
        height: Annotated[int, "number of vertical cells"] = 10

        seed: Annotated[Optional[int], "seed for the RNG"] = None

        start: Annotated[StartLocation, "location of the initial position"] = (
            StartLocation.SOUTH_WEST
        )
        rotated: Annotated[
            bool,
            ("Are mazes obtained by different start points or" " rotating SW version"),
        ] = True

        unicursive: Annotated[bool, "Single path?"] = False

        Signs = List[Sign]
        custom_classes = {
            Signs: SimpleNamespace(type_parser=str, type_name=str, default=[], action="append")
        }

        clue: Annotated[Signs, "image name for (helpful) clues"] = field(default_factory=list)

        lure: Annotated[Signs, "image name for (unhelpful) lures"] = field(default_factory=list)

        p_lure: Annotated[Optional[float], "probability of generating a lure (per-cell)"] = None

        trap: Annotated[Signs, "image name for (harmful) traps"] = field(default_factory=list)

        p_trap: Annotated[Optional[float], "probability of generating a trap (per-cell)"] = None

        def __post_init__(self):
            self._post_init(allow_unset=False)

        @staticmethod
        def _valid_dimension(d: int):
            return 2 <= d <= 100

        @staticmethod
        def _valid_probability(p: float):
            return 0 <= p <= 1

        @classmethod
        def _valid_signs(cls, signs: Signs):
            return all(cls._valid_sign(s) for s in signs)

        @staticmethod
        def _valid_sign(sign: Sign):
            return 0 < sign.value <= 1

        def _post_init(self, allow_unset: bool):
            def assert_ok(f, **kwargs):
                self._assert_field_type(f, **kwargs, allow_unset=allow_unset)

            assert_ok("width", value_tester=self._valid_dimension)
            assert_ok("height", value_tester=self._valid_dimension)

            if self.seed is None:
                self.seed = round(time.time() * 1000) % (2**31)
            else:
                assert_ok("seed", field_type=int)

            assert_ok("start")
            assert_ok("rotated")
            assert_ok("unicursive")

            if self.p_lure is not None:
                assert_ok(
                    "p_lure",
                    field_type=float,
                    value_tester=self._valid_probability,
                )

            if self.p_trap is not None:
                assert_ok(
                    "p_trap",
                    field_type=float,
                    value_tester=self._valid_probability,
                )

            if self.p_lure == 0 or not self.lure:
                if not isinstance(self.p_lure, self.Unset):
                    self.p_lure = None
                if not isinstance(self.lure, self.Unset):
                    self.lure = []

            if self.p_trap == 0 or not self.trap:
                if not isinstance(self.p_trap, self.Unset):
                    self.p_trap = None
                if not isinstance(self.p_trap, self.Unset):
                    self.trap = []

            for k in ["clue", "lure", "trap"]:
                attr = getattr(self, k)
                if isinstance(attr, list):
                    setattr(
                        self,
                        k,
                        [Sign.from_string(s) if isinstance(s, str) else s for s in attr],
                    )
                assert_ok(k, field_type=list, value_tester=self._valid_signs)

        def to_string(self) -> str:
            """Generate a string representation of this object

            :see: from_string
            """
            sep = self._FIELD_SEP
            default = Maze.BuildData()
            tokens = [f"M{self.seed}", f"{self.width}x{self.height}"]
            if self.start != default.start:
                tokens.append(self.start.shorthand())
            if self.unicursive:
                tokens.append("U")
            if not self.rotated:
                tokens.append("R")
            tokens.extend(f"C{Sign.to_string(s)}" for s in self.clue)
            if self.p_lure and self.lure:
                tokens.append(f"l{self.p_lure:.2g}".lstrip("0"))
                tokens.extend(f"L{Sign.to_string(s)}" for s in self.lure)
            if self.p_trap and self.trap:
                tokens.append(f"t{self.p_trap:.2g}".lstrip("0"))
                tokens.extend(f"T{Sign.to_string(s)}" for s in self.trap)
            return sep.join(tokens)

        @classmethod
        def from_string(cls, s, overrides: Optional["Maze.BuildData"] = None) -> "Maze.BuildData":
            """
            Parses a string to create a BuildData object.

            The complete syntax is as follows, where every element is optional
            and can be in any order:

            - M[int]: maze will use that seed
            - [int]x[int]: maze will have those dimensions
            - [SE|SW|NE|NW]: the agent will start in that corner
            - U: maze will be unicursive (without intersections)
            - R: maze is not invariant to rotation
            - C[sign]: add one helpful sign type
            - L[sign]: add one mildly deceptive sign type
            - T[sign]: add one deceptive sign type
            - l: sets the probability for lures
            - t: sets the probability for traps
            - [sign]: specification for the sign shape/value

            A sign specification contains two parameters:

            - C[shape]-[value]: sign will use the requested shape (either from
                the :meth:`~amaze.misc.resources.builtins()` or an image file
                in :meth:`~amaze.misc.resources.resources_path()`)
            - C[value]: sign will have a default shape with the given value

            Examples:
                - M16_5x10_U
                    Creates a maze 5 cells wide and 10 cells high from a random
                    number generator seeded with 16. The maze will have no
                    intersections.

                - 25x25_C1
                    Creates a maze (from a random seed) of size 25 by 25 with
                    white arrows at every intersection.

                - C1_C.5_l.25_L.25_L.2_L.3_t.5_T.75
                    Creates a maze with two types of clues with white and gray
                    arrows at half of the intersections. The remainder use a
                    trap sign (still an arrow) in light gray. A quarter of the
                    cell along the path to the target, that are *not* an
                    intersection will contain lures of varying degree of dark
                    gray (exactly one third each).

            """
            d_re = re.compile("[SN][WE]")
            s_re = re.compile("[0-9]+x[0-9]+")
            bd = cls()

            # Remove prefix (if any)
            s = s.split("__")[-1]

            for token in s.replace(cls._FIELD_SEP, " ").split():
                t, tail = token[0], token[1:]
                if t == "M":
                    bd.seed = s if (s := int(tail)) >= 0 else bd.seed
                elif t == "U":
                    bd.unicursive = True
                elif t == "R":
                    bd.rotated = False
                elif t == "C":
                    bd.clue.append(Sign.from_string(tail))
                elif t == "l":
                    bd.p_lure = float(tail) if tail else bd.p_lure
                elif t == "L":
                    bd.lure.append(Sign.from_string(tail))
                elif t == "t":
                    bd.p_trap = float(tail) if tail else bd.p_trap
                elif t == "T":
                    bd.trap.append(Sign.from_string(tail))
                elif s_re.match(token):
                    bd.width, bd.height = [int(x) for x in token.split("x")]
                elif d_re.match(token):
                    bd.start = StartLocation.from_shorthand(token)
                else:
                    raise ValueError(f"Unknown or malformed token '{token}'")

            bd._post_init(allow_unset=False)

            if overrides:
                return bd.override_with(overrides)
            else:
                return bd

        def all_rotations(self):
            """Returns a list describing the same maze with all four rotations"""
            return [
                self.where(start=s)
                for s in [
                    StartLocation.NORTH_WEST,
                    StartLocation.NORTH_EAST,
                    StartLocation.SOUTH_WEST,
                    StartLocation.SOUTH_EAST,
                ]
            ]

    Signs = BuildData.Signs

    class Direction(Enum):
        """One of the cardinal directions"""

        EAST = 0
        NORTH = 1
        WEST = 2
        SOUTH = 3

    _offsets = {
        Direction.EAST: (+1, +0),
        Direction.NORTH: (+0, +1),
        Direction.WEST: (-1, +0),
        Direction.SOUTH: (+0, -1),
    }

    _offsets_inv = {v: k for k, v in _offsets.items()}

    _inverse_dir = {
        Direction.EAST: Direction.WEST,
        Direction.NORTH: Direction.SOUTH,
        Direction.WEST: Direction.EAST,
        Direction.SOUTH: Direction.NORTH,
    }

    __private_key = object()

    PlacedSign = namedtuple("PlacedSign", ["visual_index", "solution_index", "direction", "truth"])
    """ A physically instantiated sign with a position, ..."""

    def __init__(self, data: "Maze.BuildData", key=None):
        """Private maze constructor. See `build` for the public API."""
        if key is not self.__private_key:
            raise RuntimeError("Cannot create maze directly")

        self.width, self.height = data.width, data.height
        self.walls = np.ones((data.width, data.height, 4), dtype=bool)
        self.e_start = data.start
        self.start = (0, 0)
        self.end = (self.width - 1, self.height - 1)
        self.rotated = True

        self.seed = data.seed
        self.solution: Optional[List[Tuple[int, int]]] = None
        self._intersections: int = 0

        self.signs: dict[SignType, Maze.Signs] = {
            SignType.CLUE: data.clue,
            SignType.LURE: data.lure,
            SignType.TRAP: data.trap,
        }
        self.p_lure: float = data.p_lure
        self.p_trap: float = data.p_trap
        self.signs_data: Dict[SignType, List[Maze.PlacedSign]] = {t: [] for t in SignType}

    def __repr__(self):
        return self.to_string()

    def intersections(self):
        return self._intersections

    def unicursive(self):
        return self.intersections() == 0

    def clues(self):
        return self.signs[SignType.CLUE]

    def lures(self):
        return self.signs[SignType.LURE]

    def traps(self):
        return self.signs[SignType.TRAP]

    def stats(self):
        return dict(
            size=(self.width, self.height),
            path=len(self.solution),
            intersections=self._intersections,
            clues=len(self.signs_data[SignType.CLUE]),
            lures=len(self.signs_data[SignType.LURE]),
            traps=len(self.signs_data[SignType.TRAP]),
        )

    def iter_cells(self):
        return ((i, j) for i in range(self.width) for j in range(self.height))

    def iter_solutions(self):
        return (s for s in self.solution)

    def valid(self, i, j):
        return 0 <= i < self.width and 0 <= j < self.height

    def wall(self, i: int, j: int, d: Direction):
        return self.walls[i, j, d.value]

    def wall_delta(self, i: int, j: int, di: int, dj: int):
        return self.wall(i, j, self._offsets_inv[(di, dj)])

    def direction_from_offset(self, i: int, j: int):
        return self._offsets_inv[(i, j)]

    @classproperty
    def offsets(self):
        return self._offsets

    @classproperty
    def offsets_inv(self):
        return self._offsets_inv

    def _set_wall(self, i: int, j: int, d: Direction, wall: bool):
        od = self._offsets[d]
        d_ = self._inverse_dir[d]
        i_, j_ = i + od[0], j + od[1]
        self.walls[i, j, d.value] = wall
        if self.valid(i_, j_):
            self.walls[i_, j_, d_.value] = wall

    @classmethod
    def generate(cls, data: BuildData):
        assert isinstance(
            data, Maze.BuildData
        ), f"Wrong argument type {type(data)} instead of Maze.BuildData"
        maze = Maze(data, cls.__private_key)

        def _reset_rng():
            return random.Random(maze.seed)

        rng = _reset_rng()

        w, h = maze.width, maze.height
        maze.rotated = data.rotated
        if not maze.rotated:
            maze.start = {
                StartLocation.SOUTH_WEST: (0, 0),
                StartLocation.SOUTH_EAST: (w - 1, 0),
                StartLocation.NORTH_EAST: (w - 1, h - 1),
                StartLocation.NORTH_WEST: (0, h - 1),
            }[maze.e_start]
            maze.end = (w - maze.start[0] - 1, h - maze.start[1] - 1)

        stack = [maze.start]
        visited = np.full((w, h), False)
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

        if maze.rotated and data.start is not StartLocation.SOUTH_WEST:
            # Rotate cells
            maze.walls = np.rot90(maze.walls, -data.start.value, axes=(1, 0))
            # Rotate walls inside the cells
            maze.walls = np.roll(maze.walls, data.start.value, axis=2)

            maze.width, maze.height = maze.walls.shape[:-1]
            w, h = maze.width, maze.height
            rotate = {
                StartLocation.SOUTH_WEST: lambda _i, _j: (_i, _j),
                StartLocation.SOUTH_EAST: lambda _i, _j: (w - 1 - _j, _i),
                StartLocation.NORTH_WEST: lambda _i, _j: (_j, h - 1 - _i),
                StartLocation.NORTH_EAST: lambda _i, _j: (
                    w - 1 - _i,
                    h - 1 - _j,
                ),
            }[data.start]

            # Rotate start/end
            maze.start, maze.end = rotate(*maze.start), rotate(*maze.end)

            # Rotate the solution
            maze.solution = [rotate(i, j) for i, j in maze.solution]

        # Extract intersections
        intersections = []
        for i, (c_i, c_j) in enumerate(maze.solution[1:-1]):
            # assert sum(maze.walls[c_i, c_j]) > 0, "Three-way intersection"
            if sum(maze.walls[c_i, c_j]) < 2:
                nc_i, nc_j = maze.solution[i + 2]
                d_i, d_j = nc_i - c_i, nc_j - c_j
                intersections.append((i + 1, cls._offsets_inv[(d_i, d_j)]))

        # Wall off intersections in unicursive mode
        if data.unicursive:
            for i, d in intersections:
                c0_i, c0_j = maze.solution[i - 1]
                c1_i, c1_j = maze.solution[i]
                di, dj = c0_i - c1_i, c0_j - c1_j
                dirs = cls._offsets_inv.copy()
                dirs.pop((di, dj))
                dirs.pop(cls._offsets[d])
                for _, d_ in dirs.items():
                    maze._set_wall(c1_i, c1_j, d_, True)
            intersections = []

            # Check last cell
            c_i, c_j = maze.end
            if sum(maze.walls[c_i, c_j]) < 3:
                c0_i, c0_j = maze.solution[-2]
                di, dj = c0_i - c_i, c0_j - c_j
                dirs = cls._offsets_inv.copy()
                dirs.pop((di, dj))
                for _, d_ in dirs.items():
                    maze._set_wall(c_i, c_j, d_, True)

        maze._intersections = len(intersections)

        def rng_indices(a, n):
            lst = list(islice(cycle(range(len(a))), n))
            rng.shuffle(lst)
            return lst

        # Generate clues (always but see below)
        clues_data = maze.signs_data[SignType.CLUE]
        clues = maze.signs[SignType.CLUE]
        if clues:
            rng = _reset_rng()
            clues_sign_indices = rng_indices(clues, maze._intersections)
        else:
            clues_sign_indices = [-1] * maze._intersections
        for i, (sol_index, direction) in enumerate(intersections):
            clues_data.append((clues_sign_indices[i], sol_index, direction, direction))

        # Add un-helpful signs
        lures = maze.signs[SignType.LURE]
        if maze.p_lure and lures:
            rng = _reset_rng()
            candidates = set(range(len(maze.solution) - 1)) - {i[0] for i in intersections}
            nl = round(data.p_lure * len(candidates))
            lure_indices = rng_indices(lures, nl)
            lures_data = maze.signs_data[SignType.LURE]
            dirs = list(cls._offsets_inv.values())
            if data.start is not StartLocation.SOUTH_WEST:
                dirs = list(np.roll(dirs, -data.start.value))
            for vix, six in zip(lure_indices, rng.sample(list(candidates), nl)):
                c_i, c_j = maze.solution[six]
                nc_i, nc_j = maze.solution[six + 1]
                dirs_ = dirs.copy()
                d = maze._offsets_inv[(nc_i - c_i, nc_j - c_j)]
                dirs_.remove(d)
                lures_data.append((vix, six, rng.choice(dirs_), d))

        # Transform helpful signs into harmful ones
        traps = maze.signs[SignType.TRAP]
        if maze.p_trap and traps:
            rng = _reset_rng()
            nt = round(data.p_trap * len(clues_data))
            trap_indices = sorted(rng.sample(range(maze._intersections), nt))
            trap_signs_indices = rng_indices(traps, nt)
            traps_data = maze.signs_data[SignType.TRAP]

            for i in reversed(trap_indices):
                vix, six, sign_dir, true_dir = clues_data.pop(i)
                pos = maze.solution[six]
                prev_dir = maze._offsets_inv[
                    tuple(a - b for a, b in zip(maze.solution[six - 1], maze.solution[six]))
                ]
                candidate_dirs = (
                    set(Maze.Direction)
                    - {true_dir, prev_dir}
                    - set(d for d in Maze.Direction if maze.wall(pos[0], pos[1], d))
                )
                new_d = rng.choice(list(candidate_dirs))
                traps_data.append((trap_signs_indices.pop(), six, new_d, true_dir))

        # Forget about cues if not needed
        if not clues:
            maze.signs_data[SignType.CLUE] = []

        if not data.unicursive and not clues:
            logger.warning("Mazes with intersections and no clues are" " practically unsolvable")

        return maze

    def build_data(self):
        return Maze.BuildData(
            width=self.width,
            height=self.height,
            seed=self.seed,
            start=self.e_start,
            rotated=self.rotated,
            unicursive=self.unicursive(),
            clue=self.clues(),
            p_lure=self.p_lure,
            lure=self.lures(),
            p_trap=self.p_trap,
            trap=self.traps(),
        )

    def to_string(self):
        """Provides the string representation of this maze"""
        return self.build_data().to_string()

    @classmethod
    def from_string(cls, s, overrides: Optional[BuildData] = None) -> "Maze":
        """Generate a maze from its string description.

        Optionally, specific parameters can be overridden by values set in
        the `overrides` argument.
        The full syntax is described in :meth:`.BuildData.from_string`.
        """
        return cls.generate(cls.BuildData.from_string(s, overrides))

    def all_rotations(self) -> List["Maze"]:
        """Returns all rotated versions of this maze"""
        return [self.generate(d) for d in self.build_data().all_rotations()]
