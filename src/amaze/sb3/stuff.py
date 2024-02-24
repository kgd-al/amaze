import os

import cv2

from stable_baselines3.common.env_checker import check_env

from amaze.sb3.maze_env import MazeEnv
from amaze.simu.maze import Maze
from amaze.simu.robot import Robot
from amaze.simu.types import InputType, OutputType


def check_maze_env():
    for i in InputType:
        for o in OutputType:
            env = MazeEnv(Maze.BuildData(unicursive=True),
                          Robot.BuildData(inputs=i,
                                          outputs=o,
                                          vision=36))
            check_env(env, skip_render_check=True)
            print(f"Maze({i}, {o}) OK")


def test_pygame_env():  # Manual env test
    env = MazeEnv(Maze.BuildData(seed=0, unicursive=False),
                  Robot.BuildData(inputs=InputType.CONTINUOUS,
                                  outputs=OutputType.CONTINUOUS,
                                  vision=36))

    env.reset()
    step = 0
    p_quit, done = False, False
    while not done and not p_quit:
        action = env.action_space.sample()
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"{reward=}\n{terminated=}\n{truncated=}")

        img = env.render()

        cv2.imshow("env", img)
        p_quit = (cv2.waitKey(1) == 27)

        done = terminated or truncated
        if done:
            print("Goal reached!", "reward=", reward)
        step += 1

    while not p_quit:
        p_quit = (cv2.waitKey(1) == 27)

    env.close()

    print("Testing done. Quitting")
    exit(0)
