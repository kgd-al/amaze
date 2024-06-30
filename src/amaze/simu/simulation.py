from logging import getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import Union, TypeVar, Optional, Tuple

import numpy as np
import pandas as pd

from amaze.misc.resources import SignType
from amaze.simu.maze import Maze
from amaze.simu.pos import Pos
from amaze.simu.robot import Robot
from amaze.simu.types import InputType, OutputType, Action, State
from ._inputs_evaluation import inputs_evaluation as _inputs_evaluation
from ._maze_metrics import metrics as _maze_metrics, MazeMetrics
from .controllers.base import BaseController
from ..misc import resources

logger = getLogger(__name__)

REWARDS = {
    "optimal": lambda length, dt: length,
    "compute": lambda length, dt: SimpleNamespace(
        timestep=-dt,
        backward=-1 / 10,
        collision=-2 / 10,
        finish=2 * length - 1,
    ),
}

T = TypeVar("T")
Resettable = Union[None, T]


class Simulation:
    """Serves as a bare-bones simulator for the maze-navigation environment.

    Handles all three configurations: full discrete, full continuous and
    hybrid
    """

    DiscreteVisual = Tuple[float, Maze.Direction, SignType, Maze.Direction]
    ImageVisual = np.ndarray
    NoVisual = float

    def __init__(
        self,
        maze: Resettable[Maze] = None,
        robot: Resettable[Robot.BuildData] = None,
        save_trajectory=False,
    ):

        def test_valid_set_reset(o_, s_, a_):
            assert (
                getattr(o_, s_, None) or a_
            ), f"Cannot reuse attributes from {s_} as it was never set"

        for o, s, a in [(self, "maze", maze), (self, "robot", robot)]:
            test_valid_set_reset(o, s, a)

        if maze:
            self.maze = maze

        if robot:
            self.robot = Robot(robot)

        start = Pos(*self.maze.start) + Pos(0.5, 0.5)
        self.robot.reset(start)

        self.timestep = 0
        self.last_reward = 0
        self.dt = 1 if self.data.outputs is OutputType.DISCRETE else 0.1

        sl = len(self.maze.solution)
        self.deadline = 4 * sl / self.dt
        self.rewards = REWARDS["compute"](sl, self.dt)
        self.optimal_reward = REWARDS["optimal"](sl, self.dt)
        self.stats = SimpleNamespace(steps=0, collisions=0, backsteps=0)

        self.observations = self._observations(self.data.inputs, self.data.vision)

        self.visuals = self.generate_visuals_map(self.maze, self.data.inputs, self.data.vision)

        self.trajectory, self.errors = None, None
        if save_trajectory:
            self.trajectory = pd.DataFrame(columns=["px", "py", "ax", "ay", "r"])
            if self.data.inputs is InputType.DISCRETE:
                self.errors = {t: [0, 0] for t in SignType}

        self.generate_inputs()

    @property
    def data(self):
        return self.robot.data

    def time(self):
        return self.timestep * self.dt

    def success(self):
        """Return whether the agent has reached the target"""
        return self.robot.cell() == self.maze.end

    def failure(self):
        """Return whether the agent has exceeded the deadline"""
        return self.timestep >= self.deadline

    def done(self):
        return self.success() or self.failure()

    def cumulative_reward(self):
        return self.robot.reward

    def normalized_reward(self):
        """Return the agent's cumulative reward in :math:`(-\\inf, 1]`"""
        return (
            2 * int(self.success())
            - self.dt * self.stats.steps / (len(self.maze.solution) - 1)
            - 0.01 * self.stats.backsteps
            - 0.02 * self.stats.collisions
        )

    def infos(self):
        """Returns various data about the current state of the simulation"""
        infos = dict(
            time=self.timestep,
            success=self.success(),
            failure=self.failure(),
            done=self.done(),
            pretty_reward=self.normalized_reward(),
            len=len(self.maze.solution),
            **self.stats.__dict__,
        )
        if self.errors:
            infos["errors"] = {
                t.value.lower(): (100 * v[1] / total if (total := sum(v)) > 0 else 0)
                for t, v in self.errors.items()
            }
        return infos

    def reset(self, *args, **kwargs):
        if "save_trajectory" not in kwargs:
            kwargs["save_trajectory"] = self.trajectory is not None
        self.__init__(*args, **kwargs)

    def run(self, controller):
        """Let the agent navigate in the maze until completion"""
        while not self.done():
            self.step(controller(self.observations))

    @staticmethod
    def generate_visuals_map(maze: Maze, inputs: InputType, vision: int = 15):
        visuals = np.full((maze.width, maze.height), np.nan, dtype=object)

        if inputs is InputType.CONTINUOUS:

            v = vision - 2
            images = {
                t: (resources.np_images(signs, v) if (signs := maze.signs[t]) is not None else None)
                for t in SignType
            }

            for t in SignType:
                lst, img_list = maze.signs_data[t], images[t]
                for v_index, sol_index, d, _ in lst:
                    visuals[maze.solution[sol_index]] = img_list[v_index][d.value]

        else:
            for t in SignType:
                lst, signs = maze.signs_data[t], maze.signs[t]
                for v_index, sol_index, sign_dir, true_dir in lst:
                    visuals[maze.solution[sol_index]] = (
                        signs[v_index].value,
                        sign_dir,
                        t,
                        true_dir,
                    )

        return visuals

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

        def wall(i_, j_, d_):
            return self.maze.wall(i_, j_, d_)

        def chk():
            return (
                (x - i <= r),
                (i + 1 - x <= r),
                (y - j <= r),
                (j + 1 - y <= r),
            )

        o_w, o_e, o_s, o_n = chk()

        #######################################################################
        # Simple stay-in-the cell

        if o_w:
            if wall(i, j, WEST):
                x_ = i + r

        elif i + 1 - x <= r:
            if wall(i, j, EAST):
                x_ = i + 1 - r

        if y - j <= r:
            if wall(i, j, SOUTH):
                y_ = j + r

        elif j + 1 - y <= r:
            if wall(i, j, NORTH):
                y_ = j + 1 - r

        #######################################################################
        # Literal corner cases

        o_w, o_e, o_s, o_n = chk()

        def corner_case(wall0, wall1, corner):
            if wall(*wall0) or wall(*wall1):
                cx, cy = corner
                dv = Pos(cx, cy) - new_pos
                d = dv.length()
                pen = r - d
                if d < r:
                    nonlocal x_, y_
                    dv = dv / d
                    x_ -= dv.x * pen
                    y_ -= dv.y * pen

        if o_w and o_s and i > 0 and j > 0:
            corner_case((i - 1, j, SOUTH), (i, j - 1, WEST), (i, j))

        elif o_w and o_n and i > 0 and j < h - 1:
            corner_case((i - 1, j, NORTH), (i, j + 1, WEST), (i, j + 1))

        elif o_e and o_s and i < w - 1 and j > 0:
            corner_case((i + 1, j, SOUTH), (i, j - 1, EAST), (i + 1, j))

        elif o_e and o_n and i < w - 1 and j < h - 1:
            corner_case((i + 1, j, NORTH), (i, j + 1, EAST), (i + 1, j + 1))

        #######################################################################

        new_pos = Pos(x_, y_)
        self.robot.pos = new_pos

        collision = (x != x_) + (y != y_)
        return collision

    def step(self, action: Action) -> Optional[float]:
        """Apply the requested action to the agent and return the
        corresponding reward"""
        # logger.debug(f"{'-'*80}\n-- step {self.time()}")

        reward = 0

        prev_prev_cell = self.robot.prev_cell

        pos = self.robot.pos.copy()
        prev_cell = self.robot.cell()
        if self.data.outputs == OutputType.DISCRETE:
            collision = self.__move_discrete(action)
        else:
            collision = self.__move_continuous(action)

        if collision:
            reward += self.rewards.collision
            self.stats.collisions += 1

        cell = self.robot.cell()
        if prev_cell != cell:
            self.robot.prev_cell = prev_cell

        if self.errors and (v := self._discrete_visual(self.visuals[prev_cell])):
            diff = (cell[0] - prev_cell[0], cell[1] - prev_cell[1])
            if not any(diff):
                diff = action
            d = self.maze.direction_from_offset(*diff)
            s_type = v[2]
            error = d != v[3]
            self.errors[s_type][int(error)] += 1

        reward += self.rewards.timestep
        self.stats.steps += 1

        if self.done():
            reward += self.rewards.finish

        if prev_prev_cell == self.robot.cell():
            reward += self.rewards.backward
            self.stats.backsteps += 1

        self.last_reward = reward
        self.robot.reward += reward
        self.generate_inputs()
        self.timestep += 1

        if self.trajectory is not None:
            self.trajectory.loc[len(self.trajectory)] = [*pos, *action, reward]

        return reward

    def generate_inputs(self) -> State:
        io = (self.data.inputs, self.data.outputs)
        obs: State = self.observations
        obs.fill(0)
        cell = self.robot.cell()
        prev_cell = self.robot.prev_cell

        prev_dir = None
        if prev_cell != cell:
            dx = prev_cell[0] - cell[0]
            dy = prev_cell[1] - cell[1]
            prev_dir = self.maze.direction_from_offset(dx, dy)

        walls = self.maze.walls[cell[0], cell[1]]
        visual = self.visuals[cell]
        if io == (InputType.DISCRETE, OutputType.DISCRETE):
            self._fill_discrete_visual_buffer(obs, walls, self._discrete_visual(visual), prev_dir)

        elif io == (InputType.CONTINUOUS, OutputType.DISCRETE):
            self._fill_continuous_visual_buffer(obs, walls, self._image_visual(visual), prev_dir)

        elif io == (InputType.CONTINUOUS, OutputType.CONTINUOUS):
            v = self.data.vision

            x, y = self.robot.pos
            dpx = int((x - int(x) - 0.5) * v)
            dpy = int((y - int(y) - 0.5) * v)

            if dpx == 0 and dpy == 0:
                self._fill_continuous_visual_buffer(
                    obs, walls, self._image_visual(visual), prev_dir
                )

            else:
                buffer = np.zeros((3 * v, 3 * v))
                for di, dj in [(i - 1, j - 1) for i, j in np.ndindex(3, 3)]:
                    cx, cy = cell[0] + di, cell[1] + dj
                    if not 0 <= cx <= self.maze.width - 1 or not 0 <= cy <= self.maze.height - 1:
                        continue
                    self._fill_continuous_visual_buffer(
                        buffer[
                            (-dj + 1) * v : (-dj + 2) * v,
                            (di + 1) * v : (di + 2) * v,
                        ],
                        self.maze.walls[cx, cy],
                        self._image_visual(self.visuals[(cx, cy)]),
                        prev_dir if di == 0 and dj == 0 else None,
                    )

                obs[:] = buffer[v - dpy : 2 * v - dpy, v + dpx : 2 * v + dpx]

        else:  # pragma no cover
            raise ValueError(f"Invalid I/O combination: {io}")

        return obs

    @staticmethod
    def _observations(input_type: InputType, vision: Optional[int]):
        if input_type is InputType.CONTINUOUS:
            return np.zeros((vision, vision), dtype=np.float32)
        elif input_type is InputType.DISCRETE:
            return np.zeros(8, dtype=np.float32)
        else:  # pragma no cover
            raise ValueError(f"Invalid InputType: {input_type=}")

    @staticmethod
    def _discrete_visual(visual: Union[DiscreteVisual, float]) -> Optional[DiscreteVisual]:
        return visual if not isinstance(visual, float) or not np.isnan(visual) else None

    @staticmethod
    def _image_visual(visual: Union[ImageVisual, float]) -> Optional[ImageVisual]:
        return visual if visual is not None and not np.any(np.isnan(visual)) else None

    @staticmethod
    def _fill_discrete_visual_buffer(
        buffer: State,
        walls: np.ndarray,
        visual: Optional[DiscreteVisual],
        prev_dir: Optional[Maze.Direction],
    ):
        buffer[:4] = [walls[d.value] for d in Maze.Direction]
        if prev_dir:
            buffer[prev_dir.value] = 0.5

        if visual is not None:
            buffer[4 + visual[1].value] = visual[0]

    @staticmethod
    def _fill_continuous_visual_buffer(
        buffer: State,
        walls: np.ndarray,
        visual: Optional[ImageVisual],
        prev_dir: Optional[Maze.Direction],
    ):
        # noinspection PyPep8Naming
        EAST, NORTH, WEST, SOUTH = [d for d in Maze.Direction]

        # Draw walls & corners
        for s, d in [
            (np.s_[:, -1], EAST),
            (np.s_[+0, :], NORTH),
            (np.s_[:, +0], WEST),
            (np.s_[-1, :], SOUTH),
        ]:
            buffer[s] = walls[d.value]
        for s, dc, dr in [
            ((+0, -1), NORTH, EAST),
            ((+0, +0), NORTH, WEST),
            ((-1, -1), SOUTH, EAST),
            ((-1, +0), SOUTH, WEST),
        ]:
            buffer[s] = walls[dc.value] or walls[dr.value]

        # Place cues/traps
        if visual is not None:
            buffer[1:-1, 1:-1] = visual

        # Pixel shows the previous cell
        if prev_dir:
            ix = buffer.shape[0] // 2
            s = [
                (np.s_[ix, -1]),
                (np.s_[+0, ix]),
                (np.s_[ix, +0]),
                (np.s_[-1, ix]),
            ][prev_dir.value]
            buffer[s] = 1

    @staticmethod
    def discrete_actions():
        return BaseController.discrete_actions

    @classmethod
    def compute_metrics(
        cls, maze: Maze, inputs: InputType, vision: int
    ) -> dict[Union[MazeMetrics, str]]:
        inputs = InputType.DISCRETE  # Not implemented for continuous case
        return _maze_metrics(maze, cls.generate_visuals_map(maze, inputs, vision), inputs)

    @classmethod
    def inputs_evaluation(
        cls,
        results_path: Union[Path, str],
        controller: BaseController,
        signs: dict[SignType, Maze.Signs],
        draw_inputs: bool = False,
        draw_individual_files: bool = False,
        draw_summary_file: bool = True,
        summary_file_ratio: float = 16 / 9,
    ):
        """Evaluates the provided controller on all possible inputs.

        Uses the provided lists of clues/lures/traps and tests the controller's
        capacity to take the appropriate action in all cases.
        Unlike conventional, maze-navigation evaluation for generalization
        performance evaluation, this method does not suffer from cumulative
        failure (e.g. missing one intersection may prevent reaching the goal).

        .. warning:: Only available for fully discrete and hybrid spaces

        :param results_path: Folder under which to store the resulting files.
        :param controller: Controller to evaluate.
        :param signs: Dictionary of clues/lures/traps.
        :param draw_inputs: Whether to draw inputs (without the actions)
        :param draw_individual_files: Whether to generate a separate file for
         every input/action
        :param draw_summary_file: Whether to generate a summary file for
         all input/action pairs
        :param summary_file_ratio: Width/Height ratio of the summary file
        """

        i_type, o_type = controller.input_type, controller.output_type
        if i_type is InputType.CONTINUOUS and o_type is OutputType.CONTINUOUS:
            raise ValueError(
                "Enumerating all inputs for the fully discrete"
                " case is not supported (because of combinatory"
                " explosion)."
            )

        drawer = (
            cls._fill_discrete_visual_buffer
            if i_type is InputType.DISCRETE
            else cls._fill_continuous_visual_buffer
        )

        if isinstance(results_path, str):
            results_path = Path(results_path)

        for st in SignType:  # Ensure the dictionary is well-formed
            signs.setdefault(st, [])

        return _inputs_evaluation(
            path=results_path,
            signs=signs,
            drawer=drawer,
            observations=cls._observations(i_type, controller.vision),
            controller=controller,
            draw_inputs=draw_inputs,
            draw_individual_files=draw_individual_files,
            draw_summary_file=draw_summary_file,
            summary_file_ratio=summary_file_ratio,
        )

    @classmethod
    def inputs_evaluation_from(
        cls,
        simulation: "Simulation",
        results_path: Union[Path, str],
        controller: BaseController,
        **kwargs,
    ):
        """Evaluates the provided controller on all possible inputs.

        Uses the simulation's maze to generate the list of clues/lures/traps and
        delegates to :func:`~inputs_evaluation`.

        .. warning:: Only available for fully discrete and hybrid spaces

        :param simulation: The simulation to grab maze data from.
        :param results_path: Folder under which to store the resulting files.
        :param controller: Controller to evaluate.
        :param kwargs: Additional keyword arguments.
        """

        return cls.inputs_evaluation(
            results_path=results_path,
            controller=controller,
            signs=simulation.maze.signs,
            **kwargs,
        )
