from logging import getLogger
from types import SimpleNamespace
from typing import Union, TypeVar, Optional

import numpy as np

# from amaze.simu.controllers.base import BaseController
from amaze.simu.env.maze import Maze
from amaze.simu.pos import Pos, AlignedPos
from amaze.simu.robot import Robot, InputType, OutputType, Action, State
from amaze.visu import resources

logger = getLogger(__name__)

REWARDS = {
    "optimal": lambda length, dt: length,
    "compute": lambda length, dt: SimpleNamespace(
        timestep=-dt,
        backward=-1 / 10,
        collision=-2 / 10,
        finish=2*length-1,
    )
}

T = TypeVar('T')
Resettable = Union[None, T]


class Simulation:
    def __init__(self,
                 maze: Resettable[Maze] = None,
                 robot: Resettable[Robot.BuildData] = None):

        def test_valid_set_reset(o_, s_, a_):
            assert getattr(o_, s_, None) or a_, \
                f"Cannot reuse attributes from {s_} as it was never set"
        for o, s, a in [(self, 'maze', maze), (self, 'robot', robot)]:
            test_valid_set_reset(o, s, a)

        if maze:
            self.maze = maze

        if robot:
            self.data = robot
            self.robot = Robot()

        start = Pos(*self.maze.start) + Pos(.5, .5)
        self.robot.reset(start)

        self.timestep = 0
        self.dt = 1 if self.data.outputs is OutputType.DISCRETE else .1

        sl = len(self.maze.solution)
        self.deadline = 4 * sl / self.dt
        self.rewards = REWARDS["compute"](sl, self.dt)
        self.optimal_reward = REWARDS["optimal"](sl, self.dt)
        self.stats = SimpleNamespace(
            steps=0, collisions=0, backsteps=0
        )

        self.cues = None
        self.traps = None

        if self.data.inputs is InputType.CONTINUOUS:
            self.observations = np.zeros((self.data.vision, self.data.vision),
                                         dtype=np.float32)
        else:
            self.observations = np.zeros(8, dtype=np.float32)

        self.visuals = np.full((self.maze.width, self.maze.height), np.nan,
                               dtype=object)

        if self.data.inputs is InputType.CONTINUOUS:

            v = self.data.vision
            clues = resources.np_images(self.maze.clues, v - 2) \
                if self.maze.clues is not None else None
            lures = resources.np_images(self.maze.lures, v - 2) \
                if self.maze.clues is not None else None
            traps = resources.np_images(self.maze.traps, v - 2) \
                if self.maze.traps is not None else None

            for lst, img_lst in [(self.maze.clues_data, clues),
                                 (self.maze.lures_data, lures),
                                 (self.maze.traps_data, traps)]:
                if lst is not None and img_lst is not None:
                    for v_index, sol_index, d in lst:
                        self.visuals[self.maze.solution[sol_index]] = \
                            img_lst[v_index][d.value]

        elif self.data.inputs is InputType.DISCRETE:
            for lst, signs in [(self.maze.clues_data, self.maze.clues),
                               (self.maze.lures_data, self.maze.lures),
                               (self.maze.traps_data, self.maze.traps)]:
                if lst is not None:
                    for v_index, sol_index, d in lst:
                        self.visuals[self.maze.solution[sol_index]] = \
                            (signs[v_index].value, d)

        self.generate_inputs()

    def time(self):
        return self.timestep * self.dt

    def reset(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    def __move_discrete(self, action: Action) -> bool:
        x, y = self.robot.cell()
        if self.maze.wall_delta(x, y, action[0], action[1]):
            return True
        else:
            self.robot.pos += action
            return False

    def __move_continuous(self, action: Action) -> bool:
        # noinspection PyPep8Naming
        EAST, NORTH, WEST, SOUTH = [d for d in Maze.Direction]
        w, h = self.maze.width, self.maze.height

        x, y = new_pos = self.robot.next_position(action, self.dt)
        x_, y_ = x, y
        i, j = new_pos.aligned()
        r = self.robot.RADIUS

        def wall(i_, j_, d_): return self.maze.wall(i_, j_, d_)

        def chk(): return (x - i <= r), (i + 1 - x <= r),\
                          (y - j <= r), (j + 1 - y <= r)

        o_w, o_e, o_s, o_n = chk()

        #######################################################################
        # Simple stay-in-the cell sb3

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

        new_pos = Pos(x_, y_)
        self.robot.pos = new_pos

        collision = ((x != x_) + (y != y_))
        return collision

    def step(self, action: Action):
        # logger.debug(f"{'-'*80}\n-- step {self.time()}")

        if self.data.control == "KEYBOARD" and \
                not action and not self.robot.vel:
            return

        reward = 0

        prev_prev_cell = self.robot.prev_cell

        prev_cell = self.robot.cell()
        if self.data.outputs == OutputType.DISCRETE:
            collision = self.__move_discrete(action)
        else:
            collision = self.__move_continuous(action)

        if collision:
            reward += self.rewards.collision
            self.stats.collisions += 1

        if prev_cell != self.robot.cell():
            self.robot.prev_cell = prev_cell

        reward += self.rewards.timestep
        self.stats.steps += 1

        if self.done():
            reward += self.rewards.finish

        if prev_prev_cell == self.robot.cell():
            reward += self.rewards.backward
            self.stats.backsteps += 1

        self.robot.reward += reward
        self.generate_inputs()
        self.timestep += 1

        return reward

    def generate_inputs(self) -> State:
        i: State = self.observations
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
                i[prev_dir.value] = .5

            if d := self.visuals[cell]:
                if not isinstance(d, float) or not np.isnan(d):
                    i[4+d[1].value] = d[0]

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

    def _fill_visual_buffer(self, buffer: np.ndarray,
                            cell: AlignedPos,
                            prev_dir: Optional[Maze.Direction]):
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

    def robot_dict(self) -> Optional[dict]:
        if r := self.robot:
            return dict(pos=r.pos, vel=r.vel, acc=r.acc)
        else:
            return None

    def success(self):
        return self.robot.cell() == self.maze.end

    def failure(self):
        return self.timestep >= self.deadline

    def done(self):
        return self.success() or self.failure()

    def infos(self):
        return dict(
            time=self.timestep,
            success=self.success(),
            failure=self.failure(),
            done=self.done(),
            pretty_reward=
            2 * int(self.success())
            - self.stats.steps / (len(self.maze.solution) - 1)
            - .01 * self.stats.backsteps
            - .02 * self.stats.collisions,
            len=len(self.maze.solution),
            **self.stats.__dict__
        )

    @staticmethod
    def discrete_actions():
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]
