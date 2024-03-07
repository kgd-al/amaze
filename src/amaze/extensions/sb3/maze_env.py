""" SB3 wrapper for the maze environment """

import logging
from typing import Optional, List

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter
from gymnasium import spaces, Env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from amaze import application
from amaze.extensions.sb3.utils import IOMapper
from amaze.extensions.sb3 import CV2QTGuard
from amaze.simu.maze import Maze
from amaze.simu.robot import Robot
from amaze.simu.simulation import Simulation
from amaze.simu.types import InputType, OutputType
from amaze.visu.widgets.maze import MazeWidget

logger = logging.getLogger(__name__)


def make_vec_maze_env(mazes: List[Maze.BuildData],
                      robot: Robot.BuildData,
                      seed, **kwargs):
    """ Encapsulates the creation of a vectorized environment """

    mazes = [m for m in mazes]

    def env_fn():
        env = MazeEnv(mazes.pop(0), robot, **kwargs)
        check_env(env)
        env.reset(full_reset=True)
        return env

    return make_vec_env(env_fn, n_envs=len(mazes), seed=seed)


def env_method(env, method: str, *args, **kwargs):
    """ Calls a given function, with specified arguments, on each underlying
    environments """
    return [getattr(e.unwrapped, method)(*args, **kwargs) for e in env.envs]


def env_attr(env, attr: str):
    """ Returns the requested attribute from each underlying environments """
    return [getattr(e.unwrapped, attr) for e in env.envs]


class MazeEnv(Env):
    """ AMaze wrapper for the stable baselines 3 library
    """
    metadata = dict(
        render_modes=["human", "rgb_array"],
        render_fps=30,
        min_resolution=256
    )

    def __init__(self, maze: Maze.BuildData, robot: Robot.BuildData,
                 log_trajectory: bool = False):
        """ Built with maze data and robot data

        :param ~amaze.simu.maze.Maze.BuildData maze: maze data
        :param ~amaze.simu.robot.Robot.BuildData robot: agent data
        """
        super().__init__()
        self.render_mode = "rgb_array"

        self.name = maze.to_string()

        self._simulation = Simulation(Maze.generate(maze), robot,
                                      save_trajectory=log_trajectory)
        _pretty_rewards = \
            ', '.join(f'{k}: {v:.2g}'
                      for k, v in self._simulation.rewards.__dict__.items())
        logger.debug(f"Creating MazeEnv with"
                     f"\n       {self.name}"
                     f"\n maze: {maze}"
                     f"\nrobot: {robot}"
                     f"\nrewards: [{_pretty_rewards}]"
                     f"\n{log_trajectory=}")

        self.observation_type = robot.inputs
        self.observation_space = {
            InputType.DISCRETE: spaces.Box(low=-1, high=1, shape=(8, ),
                                           dtype=np.float32),
            InputType.CONTINUOUS:
                spaces.Box(low=0, high=255,
                           shape=(1, robot.vision, robot.vision),
                           dtype=np.uint8)
        }[robot.inputs]

        self.action_type = robot.outputs
        self.action_space = {
            OutputType.DISCRETE: spaces.Discrete(4),
            OutputType.CONTINUOUS:
                spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        }[robot.outputs]

        self.mapper = IOMapper(observation_space=self.observation_space,
                               action_space=self.action_space)

        self.widget, self.app = None, None

        self.prev_trajectory = None

        self.last_infos = None
        self.length = len(self._simulation.maze.solution)

        self.resets = 0

    def reset(self, seed=None, options=None, full_reset=False):
        """Stub """
        self.last_infos = self.infos()
        if self._simulation.trajectory is not None:
            self.prev_trajectory = self._simulation.trajectory.copy(True)

        super().reset(seed=seed)
        self._simulation.reset()

        maze_str = self._simulation.maze.to_string()
        if full_reset:
            self.resets = 0
            logger.debug(f"Initial reset for {maze_str}")
        else:
            self.resets += 1
            logger.debug(f"Reset {self.resets} for {maze_str}")

        return self._observations(), self.infos()

    def step(self, action):
        """ Stub docstring
        """
        vec_action = self.mapper.map_action(action)

        reward = self._simulation.step(vec_action)
        observation = self._observations()
        terminated = self._simulation.success()
        truncated = self._simulation.failure()
        info = self._simulation.infos()

        # done = terminated or truncated
        # logger.debug(f"Step {self._simulation.timestep:03d} ({done=})"
        #              f" for {self._simulation.maze.to_string()}")

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """ Stub """
        with CV2QTGuard():  # Using Qt in CV2 context -> Protect
            s = 256

            if self.widget is None:
                self.widget = self._create_widget(size=s, show_robot=True)

            img = QImage(s, s, QImage.Format_RGB888)
            img.fill(Qt.white)

            painter = QPainter(img)
            self.widget.render(painter)
            painter.end()

            return self._qimage_to_numpy(img)

    def close(self):
        """ Stub """
        pass

    def name(self): return self.name
    def atomic_rewards(self): return self._simulation.rewards

    def optimal_reward(self):
        """ Return the cumulative reward for an agent following an optimal
        trajectory"""
        return self._simulation.optimal_reward

    def maximal_duration(self): return self._simulation.deadline

    def io_types(self): return (self._simulation.data.inputs,
                                self._simulation.data.outputs)

    def log_trajectory(self, do_log: bool):
        self._simulation.reset(save_trajectory=do_log)

    def plot_trajectory(self, cb_side: int = 0, verbose: bool = True,
                        square: bool = False) -> np.ndarray:
        with CV2QTGuard():
            _ = application()
            plot = MazeWidget.plot_trajectory(
                simulation=self._simulation,
                size=256, trajectory=self.prev_trajectory,
                config=dict(
                    solution=True,
                    robot=False,
                    dark=True
                ),
                side=cb_side,
                verbose=verbose,
                square=square,
                img_format=QImage.Format_RGBA8888,
                path=None
            )
            img = self._qimage_to_numpy(plot)
        return img

    def _create_widget(self, size, show_robot=True):
        if self.widget:
            return self.widget

        app = QtWidgets.QApplication.instance()
        if app is None:
            # logger.debug("Creating qt app")
            self.app = QtWidgets.QApplication([])

        # logger.debug("Creating qt widget")

        self.widget = MazeWidget(
            simulation=self._simulation,
            resolution=self._simulation.data.vision,
            width=size
        )
        self.widget.update_config(
            robot=show_robot, solution=True, dark=True)
        return self.widget

    @staticmethod
    def _qimage_to_numpy(img: QImage) -> np.ndarray:
        w, h, d = img.width(), img.height(), img.depth() // 8
        b = img.constBits().asstring(img.byteCount())
        bw = img.bytesPerLine() // d
        return np.ndarray(
            shape=(h, bw, d), buffer=b, dtype=np.uint8)[:, :w]

    def _observations(self):
        return self.mapper.map_observation(self._simulation.observations)

    def infos(self): return self._simulation.infos()
