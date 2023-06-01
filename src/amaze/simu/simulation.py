from dataclasses import dataclass, fields
from enum import Enum, auto
from logging import getLogger
from random import Random
from types import SimpleNamespace
from typing import Tuple, Optional, Annotated
from PIL import Image

import numpy as np
from PyQt5.QtCore import QObject, Qt, QEvent
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication

from amaze.simu.env.maze import Maze
from amaze.visu import resources

logger = getLogger(__name__)


class Pos:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        try:
            return self.x == other[0] and self.y == other[1]
        except:
            return False

    def __iadd__(self, other):
        self.x += other[0]
        self.y += other[1]
        return self

    def __add__(self, other):
        return Pos(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Pos(self.x - other.x, self.y - other.y)

    def __getitem__(self, i):
        if i < 0 or 1 < i:
            raise IndexError
        return self.x if i == 0 else self.y

    def aligned(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


class InputType(Enum):
    DISCRETE = auto()
    CONTINUOUS = auto()


class OutputType(Enum):
    DISCRETE = auto()
    CONTINUOUS = auto()


class ControlType(Enum):
    AUTONOMOUS = auto()
    RANDOM = auto()
    KEYBOARD = auto()


class RandomController:
    def __init__(self):
        self.rng = Random(0)
        self.stack = []
        self.visited = set()
        self.last_pos = None

    def __call__(self, pos, _):
        ci, cj = pos = pos.aligned()
        self.visited.add(pos)

        if self.last_pos and self.last_pos == pos:
            self.stack.pop()  # Cancel last move

        neighbors = []
        for d, (di, dj) in Maze._offsets.items():
            i, j = ci + di, cj + dj
            if (i, j) not in self.visited:
                neighbors.append((d, (i, j), (di, dj)))

        self.last_pos = pos
        if len(neighbors) > 0:
            d_chosen, chosen, delta = self.rng.choice(neighbors)
            self.stack.append((-delta[0], -delta[1]))

            return delta
        else:
            return self.stack.pop()


class KeyboardController(QObject):
    def __init__(self):
        super().__init__()
        QApplication.instance().installEventFilter(self)
        self.actions = {
            Qt.Key_Right: (1, 0),
            Qt.Key_Up: (0, 1),
            Qt.Key_Left: (-1, 0),
            Qt.Key_Down: (0, -1),
        }
        self.current_action = None

    def eventFilter(self, obj: 'QObject', e: 'QEvent') -> bool:
        if not e.type() == QEvent.KeyPress:
            return False

        if e.key() in self.actions:
            self.current_action = self.actions[e.key()]
            return True
        return False

    def __call__(self, *_):
        if self.current_action is None:
            return 0, 0
        else:
            a = self.current_action
            self.current_action = None
            return a


class Robot:
    @dataclass
    class BuildData:
        vision: Annotated[int, "agent vision size"] = 10
        inputs: InputType = InputType.DISCRETE
        outputs: OutputType = OutputType.DISCRETE
        control: ControlType = ControlType.KEYBOARD

        @classmethod
        def from_argparse(cls, namespace):
            logger.warning(f"argparse extraction Not implemented for {cls}")
            data = cls()
            for field in fields(cls):
                setattr(data, field.name, None)
            return data

    def __init__(self, pos: Pos):
        self.pos = Pos(*pos)
        self.reward = 0

        self.inputs = None
        self.controller = None

        self.acc = (0, 0)

        self.prev_pos = pos.aligned()

    def set_input_buffer(self, buffer):
        self.inputs = buffer

    def step(self) -> Pos:
        return self.controller(self.pos, self.inputs)

        # return Pos(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))


def _compute_rewards(length):
    return SimpleNamespace(
        timestep=-.001/length,
        collision=-.05/length,
        finish=1
    )


class Simulation:
    def __init__(self,
                 maze: Optional[Maze] = None,
                 data: Optional[Robot.BuildData] = None):
        self.maze = maze
        self.robot = None
        if maze is not None:
            start = Pos(*maze.start) + Pos(.5, .5)
            self.robot = Robot(start)

            sl = len(self.maze.solution)
            self.deadline = 2 * sl
            self.rewards = _compute_rewards(sl)

        self.data = None
        self.timestep = 0
        self.dt = 1

        self.data = data
        self.cues = None
        self.traps = None
        if data is not None:
            if data.inputs is InputType.CONTINUOUS:
                inputs = np.zeros((data.vision, data.vision), dtype=float)
            else:
                inputs = np.zeros(4)
            self.robot.set_input_buffer(inputs)

            if data.control is ControlType.RANDOM:
                self.robot.controller = RandomController()
            elif data.control is ControlType.KEYBOARD:
                self.robot.controller = KeyboardController()
            else:
                raise NotImplementedError

        self.visuals = None
        if maze is not None and data is not None \
                and data.inputs is InputType.CONTINUOUS:
            self.visuals = np.full((maze.width, maze.height), None,
                                   dtype=object)

            v = data.vision
            cues = resources.np_images(self.maze.cue, v - 2) \
                if self.maze.cue is not None else None
            traps = resources.np_images(self.maze.trap, v - 2) \
                if self.maze.trap is not None else None

            for lst, img_lst in [(maze.intersections, cues),
                                 (maze.traps, traps)]:
                if lst is not None and img_lst is not None:
                    for i, d in lst:
                        self.visuals[maze.solution[i]] = img_lst[d.value]

        if self.robot:
            self.generate_inputs()

    def time(self):
        return self.timestep * self.dt

    def reset(self, *args):
        self.__init__(*args)

    def step(self):
        # logger.debug(f"{'-'*80}\n-- step {self.time()}")

        action = self.robot.step()

        if self.data.control == ControlType.KEYBOARD and action == (0, 0):
            return

        if abs(action[0]) > abs(action[1]):
            action_ = (action[0], 0)
        else:
            action_ = (0, action[1])

        if action != (0, 0):
            if self.maze.wall_delta(*self.robot.pos.aligned(), action_):
                self.robot.reward += self.rewards.collision
            else:
                prev_pos = self.robot.pos.aligned()
                self.robot.pos += action_

                if prev_pos != self.robot.pos.aligned():
                    self.robot.prev_pos = prev_pos

        if self.done():
            self.robot.reward += self.rewards.finish
        else:
            self.robot.reward += self.rewards.timestep

        self.timestep += 1
        self.generate_inputs()

    def generate_inputs(self):
        i = self.robot.inputs
        i.fill(0)
        pos = self.robot.pos.aligned()
        def _wall(d): return self.maze.wall(*pos, d)
        if self.data.inputs is InputType.DISCRETE:
            i[:] = [_wall(d) for d in Maze.Direction]
        else:

            # Draw walls & corners
            D = Maze.Direction
            w = [_wall(d) for d in D]
            for s, d in [(np.s_[:, -1], D.EAST),
                         (np.s_[+0, :], D.NORTH),
                         (np.s_[:, +0], D.WEST),
                         (np.s_[-1, :], D.SOUTH)]:
                i[s] = w[d.value]
            for s, dc, dr in [((+0, -1), D.NORTH, D.EAST),
                              ((+0, +0), D.NORTH, D.WEST),
                              ((-1, -1), D.SOUTH, D.EAST),
                              ((-1, +0), D.SOUTH, D.WEST)]:
                i[s] = (w[dc.value] or w[dr.value])

            # Place cues/traps
            if self.visuals is not None and \
                    (v := self.visuals[pos]) is not None:
                i[1:-1, 1:-1] = v

            # Pixel shows the previous cell
            if self.robot.prev_pos != pos:
                dx = self.robot.prev_pos[0] - pos[0]
                dy = self.robot.prev_pos[1] - pos[1]
                ix = i.shape[0]//2
                d = self.maze.direction_from_offset(dx, dy)
                s = [(np.s_[ix, -1]),
                     (np.s_[+0, ix]),
                     (np.s_[ix, +0]),
                     (np.s_[-1, ix])][d.value]
                i[s] = 1


    def done(self):
        return self.robot.pos.aligned() == self.maze.end \
            or self.timestep >= self.deadline
