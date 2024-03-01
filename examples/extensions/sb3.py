import pprint

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure

from amaze import Maze, Robot
from amaze.extensions.sb3.callbacks import TensorboardCallback
from amaze.extensions.sb3.controller import SB3Controller, wrap
from amaze.extensions.sb3.maze_env import make_vec_maze_env, env_method, env_attr, MazeEnv

FOLDER = "tmp/demos/sb3"
SEED = 16
BUDGET = 100000
VERBOSE = False


def main():
    train_mazes = Maze.BuildData.from_string("M16_10x10_C1").all_rotations()
    eval_mazes = [d.where(seed=14) for d in train_mazes]
    robot = Robot.BuildData.from_string("DD")

    train_env = make_vec_maze_env(train_mazes, robot, SEED)
    eval_env = make_vec_maze_env(eval_mazes, robot, SEED, log_trajectory=True)

    optimal_reward = (sum(env_method(eval_env, "optimal_reward"))
                      / len(eval_mazes))
    tb_callback = TensorboardCallback(
        log_trajectory_every=1,  # Eval callback (below)
        max_timestep=BUDGET
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=FOLDER, log_path=FOLDER,
        eval_freq=BUDGET//(10*len(eval_mazes)), verbose=0,
        n_eval_episodes=len(eval_mazes),
        callback_after_eval=tb_callback,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=optimal_reward, verbose=1)
    )

    model = SB3Controller(
        PPO, policy="MlpPolicy", env=train_env, seed=SEED, learning_rate=1e-3)
    print(model, type(model))

    print("== Starting", "="*68)
    model.set_logger(configure(FOLDER, ["csv", "tensorboard"]))
    model.learn(BUDGET, callback=eval_callback, progress_bar=True)

    tb_callback.log_step(True)
    print("="*80)


if __name__ == "__main__":
    main()
