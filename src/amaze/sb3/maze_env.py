import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter
from gymnasium import spaces, Env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from amaze.simu.env.maze import Maze
from amaze.simu.robot import Robot, OutputType, InputType
from amaze.simu.simulation import Simulation
from amaze.sb3.utils import CV2QTGuard, IOMapper
from amaze.visu.widgets.maze import MazeWidget

logger = logging.getLogger(__name__)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        print("="*80)
        print("== CustomCNN")
        print("="*80)

        print(n_input_channels)
        print(observation_space)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        exit(42)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class MazeEnv(Env):
    metadata = dict(
        render_modes=["human", "rgb_array"],
        render_fps=30,
        min_resolution=256
    )

    def __init__(self, maze: Maze.BuildData, robot: Robot.BuildData,
                 log_trajectory: bool = False):
        super().__init__()
        self.render_mode = "rgb_array"

        self.name = Maze.bd_to_string(maze)

        self._simulation = Simulation(Maze.generate(maze), robot)
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

        self.trajectory = pd.DataFrame(columns=["px", "py", "ax", "ay", "r"])\
            if log_trajectory else None
        self.prev_trajectory = None

        self.last_infos = None

        self.length = len(self._simulation.maze.solution)

        self.resets = 0

    def reset(self, seed=None, options=None):
        self.last_infos = self.infos()

        super().reset(seed=seed)
        self._simulation.reset()
        self.resets += 1
        logger.debug(f"Reset {self.resets}"
                     f" for {self._simulation.maze.to_string()}")

        if self.trajectory is not None:
            self.prev_trajectory = self.trajectory.copy(True)
            self.trajectory = pd.DataFrame(columns=self.trajectory.columns)

        return self._observations(), self.infos()

    def step(self, action):
        pos = self._simulation.robot.pos.copy()
        vec_action = self.mapper.map_action(action)

        reward = self._simulation.step(vec_action)
        observation = self._observations()
        terminated = self._simulation.success()
        truncated = self._simulation.failure()
        info = self._simulation.infos()

        if self.trajectory is not None:
            self.trajectory.loc[len(self.trajectory)] = \
                [*pos, *vec_action, reward]

        # done = terminated or truncated
        # logger.debug(f"Step {self._simulation.timestep:03d} ({done=})"
        #              f" for {self._simulation.maze.to_string()}")

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
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
        # if self.app is not None:
        #     logger.debug("Closing Qt")
        #     self.app.
        pass

    def name(self): return self.name
    def atomic_rewards(self): return self._simulation.rewards
    def optimal_reward(self): return self._simulation.optimal_reward
    def duration(self): return self.length

    def io_types(self): return (self._simulation.data.inputs,
                                self._simulation.data.outputs)

    def plot_trajectory(self) -> np.ndarray:
        with CV2QTGuard():
            return self._qimage_to_numpy(
                self._create_widget(256, show_robot=False)
                .plot_trajectory(self.prev_trajectory,
                                 img_format=QImage.Format_RGBA8888))

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
            size=(size, size)
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