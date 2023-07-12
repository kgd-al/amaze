from logging import getLogger
from types import SimpleNamespace
from typing import Union, TypeVar

import numpy as np

from amaze.simu.controllers.base import BaseController
from amaze.simu.env.maze import Maze
from amaze.simu.pos import Pos
from amaze.simu.robot import Robot, InputType, OutputType, Action, State
from amaze.visu import resources

logger = getLogger(__name__)


def _optimal_reward(): return 1


def _compute_rewards(length, dt):
    return SimpleNamespace(
        timestep=-dt/(length-1),
        backward=-1/100,
        collision=-2/100,
        finish=2,
    )


T = TypeVar('T')
Resettable = Union[None, T]


class Simulation:
    def __init__(self,
                 maze: Resettable[Maze] = None,
                 robot: Resettable[Robot.BuildData] = None,
                 controller: Resettable[BaseController] = None):

        def test_valid_set_reset(o_, s_, a_):
            assert getattr(o_, s_, None) or a_, \
                f"Cannot reuse attributes from {s_} as it was never set"
        for o, s, a in [(self, 'maze', maze), (self, 'robot', robot),
                        (getattr(self, 'robot', None),
                         'controller', controller)]:
            test_valid_set_reset(o, s, a)

        if maze:
            self.maze = maze

        if robot:
            self.data = robot
            self.robot = Robot()

        if controller:
            self.robot.controller = controller
        self.robot.controller.reset()

        start = Pos(*self.maze.start) + Pos(.5, .5)
        self.robot.reset(start)

        self.timestep = 0
        self.dt = 1 if self.data.outputs is OutputType.DISCRETE else .1

        sl = len(self.maze.solution)
        self.deadline = 4 * sl / self.dt
        self.rewards = _compute_rewards(sl, self.dt)

        self.cues = None
        self.traps = None

        if self.data.inputs is InputType.CONTINUOUS:
            inputs = np.zeros((self.data.vision, self.data.vision),
                              dtype=float)
        else:
            inputs = np.zeros(8)
        self.robot.set_input_buffer(inputs)

        self.robot.set_dt(self.dt)

        self.visuals = np.full((self.maze.width, self.maze.height), np.nan,
                               dtype=object)

        if self.data.inputs is InputType.CONTINUOUS:

            v = self.data.vision
            cues = resources.np_images(self.maze.cue, v - 2) \
                if self.maze.cue is not None else None
            traps = resources.np_images(self.maze.trap, v - 2) \
                if self.maze.trap is not None else None

            for lst, img_lst in [(self.maze.intersections, cues),
                                 (self.maze.traps, traps)]:
                if lst is not None and img_lst is not None:
                    for i, d in lst:
                        self.visuals[self.maze.solution[i]] = img_lst[d.value]

        elif self.data.inputs is InputType.DISCRETE:
            for lst in [self.maze.intersections, self.maze.traps]:
                if lst is not None:
                    for i, d in lst:
                        self.visuals[self.maze.solution[i]] = d

        self.generate_inputs()

    def time(self):
        return self.timestep * self.dt

    def reset(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    def action(self) -> Action:
        action = self.robot.step()

        if self.data.outputs == OutputType.DISCRETE:
            if abs(action[0]) > abs(action[1]):
                action = (action[0], 0)
            else:
                action = (0, action[1])

        return action

    def _discrete_collision_detection(self, action: Action):
        x, y = self.robot.cell()
        if self.maze.wall_delta(x, y, action[0], action[1]):
            return self.rewards.collision
        else:
            self.robot.pos += action
            return 0

    def _continuous_collision_detection(self, action: Action):
        # noinspection PyPep8Naming
        EAST, NORTH, WEST, SOUTH = [d for d in Maze.Direction]
        w, h = self.maze.width, self.maze.height
        pos = self.robot.pos
        x, y = new_pos = pos.copy() + action
        x_, y_ = x, y
        i, j = new_pos.aligned()
        r = self.robot.RADIUS

        def wall(i_, j_, d_): return self.maze.wall(i_, j_, d_)
        def chk(): return (x - i <= r), (i + 1 - x <= r), (y - j <= r), (j + 1 - y <= r)

        o_w, o_e, o_s, o_n = chk()

        #######################################################################
        # Simple stay-in-the cell tests

        if o_w:
            if wall(i, j, WEST):
                x_ = i+r

        elif i + 1 - x <= r:
            if wall(i, j, EAST):
                x_ = i+1-r

        if y - j <= r:
            if wall(i, j, SOUTH):
                y_ = j+r

        elif j + 1 - y <= r:
            if wall(i, j, NORTH):
                y_ = j+1-r

        #######################################################################
        # Literal corner cases

        o_w, o_e, o_s, o_n = chk()

        def corner_case(wall0, wall1, corner):
            if wall(*wall0) or wall(*wall1):
                cx, cy = corner
                dv = (Pos(cx, cy) - new_pos)
                d = dv.length()
                pen = r - d
                if d < r:
                    nonlocal x_, y_
                    dv = dv / d
                    x_ -= dv.x * pen
                    y_ -= dv.y * pen

        if o_w and o_s and i > 0 and j > 0:
            corner_case((i - 1, j, SOUTH), (i, j - 1, WEST), (i, j))

        elif o_w and o_n and i > 0 and j < h-1:
            corner_case((i - 1, j, NORTH), (i, j + 1, WEST), (i, j+1))

        elif o_e and o_s and i < w-1 and j > 0:
            corner_case((i + 1, j, SOUTH), (i, j - 1, EAST), (i+1, j))

        elif o_e and o_n and i < w-1 and j < h-1:
            corner_case((i + 1, j, NORTH), (i, j + 1, EAST), (i+1, j+1))

        #######################################################################
        # Also check next cells

        new_pos = Pos(x_, y_)
        self.robot.pos = new_pos

        return ((x != x_) + (y != y_)) * self.rewards.collision

    def take_action(self, action: Action) -> float:
        reward = 0

        prev_prev_cell = self.robot.prev_cell

        if action != (0, 0):
            prev_cell = self.robot.cell()
            if self.data.outputs == OutputType.DISCRETE:
                reward += self._discrete_collision_detection(action)
            else:
                reward += self._continuous_collision_detection(action)

            if prev_cell != self.robot.cell():
                self.robot.prev_cell = prev_cell

        reward += self.rewards.timestep

        if self.done():
            reward += self.rewards.finish

        if prev_prev_cell == self.robot.cell():
            reward += self.rewards.backward

        self.robot.reward += reward
        self.generate_inputs()
        self.timestep += 1

        return reward

    def step(self):
        # logger.debug(f"{'-'*80}\n-- step {self.time()}")

        action = self.action()

        if self.data.control == "KEYBOARD" and action == (0, 0):
            return

        self.take_action(action)

    def generate_inputs(self) -> State:
        i: State = self.robot.inputs
        i.fill(0)
        cell = self.robot.cell()
        prev_cell = self.robot.prev_cell

        prev_dir = None
        if prev_cell != cell:
            dx = prev_cell[0] - cell[0]
            dy = prev_cell[1] - cell[1]
            prev_dir = self.maze.direction_from_offset(dx, dy)

        if self.data.inputs is InputType.DISCRETE:
            i[:4] = [self.maze.wall(cell[0], cell[1], d) for d in Maze.Direction]
            if prev_dir:
                i[prev_dir.value] = -1

            if isinstance(d := self.visuals[cell], Maze.Direction):
                i[4+d.value] = 1

        else:

            if self.data.outputs is OutputType.DISCRETE:
                self._fill_visual_buffer(i, cell, prev_dir)
            else:
                v = self.data.vision

                x, y = self.robot.pos
                dpx = int((x - int(x) - .5) * v)
                dpy = int((y - int(y) - .5) * v)

                if dpx == 0 and dpy == 0:
                    self._fill_visual_buffer(i, cell, prev_dir)

                else:
                    buffer = np.zeros((3*v, 3*v))
                    for di, dj in [(i - 1, j - 1) for i, j in np.ndindex(3, 3)]:
                        cx, cy = cell[0] + di, cell[1] + dj
                        if not 0 <= cx <= self.maze.width - 1 \
                                or not 0 <= cy <= self.maze.height - 1:
                            continue
                        self._fill_visual_buffer(
                            buffer[(-dj+1)*v:(-dj+2)*v, (di+1)*v:(di+2)*v],
                            (cx, cy),
                            prev_dir if di == 0 and dj == 0 else None)

                    i[:] = buffer[v-dpy:2*v-dpy, v+dpx:2*v+dpx]

        return i

    def _fill_visual_buffer(self, buffer, cell, prev_dir):
        # noinspection PyPep8Naming
        EAST, NORTH, WEST, SOUTH = [d for d in Maze.Direction]
        def _wall(d): return self.maze.wall(*cell, d)

        # Draw walls & corners
        w = [_wall(d) for d in Maze.Direction]
        for s, d in [(np.s_[:, -1], EAST),
                     (np.s_[+0, :], NORTH),
                     (np.s_[:, +0], WEST),
                     (np.s_[-1, :], SOUTH)]:
            buffer[s] = w[d.value]
        for s, dc, dr in [((+0, -1), NORTH, EAST),
                          ((+0, +0), NORTH, WEST),
                          ((-1, -1), SOUTH, EAST),
                          ((-1, +0), SOUTH, WEST)]:
            buffer[s] = (w[dc.value] or w[dr.value])

        # Place cues/traps
        if self.visuals is not None and \
                not np.any(np.isnan(v := self.visuals[cell])):
            buffer[1:-1, 1:-1] = v

        # Pixel shows the previous cell
        if prev_dir:
            ix = self.data.vision // 2
            s = [(np.s_[ix, -1]),
                 (np.s_[+0, ix]),
                 (np.s_[ix, +0]),
                 (np.s_[-1, ix])][prev_dir.value]
            buffer[s] = 1

    @staticmethod
    def optimal_reward():
        return _optimal_reward()

    def success(self):
        return self.robot.cell() == self.maze.end

    def done(self):
        return self.success() or self.timestep >= self.deadline
