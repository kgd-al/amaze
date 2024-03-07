import math
import pathlib
import random
import shutil

from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.logger import configure

from amaze import Maze, Robot, Simulation, Sign, amaze_main
from amaze.extensions.sb3 import (make_vec_maze_env, env_method,
                                  load_sb3_controller, PPO,
                                  TensorboardCallback, sb3_controller, CV2QTGuard)

FOLDER = "tmp/demos/sb3"
BEST = f"{FOLDER}/best_model.zip"
SEED = 0
BUDGET = 100000
VERBOSE = False

TRAIN_MAZE = "M14_20x20_C1"
TEST_SEED = 18


def train():
    train_mazes = Maze.BuildData.from_string(TRAIN_MAZE).all_rotations()
    eval_mazes = [d.where(seed=TEST_SEED) for d in train_mazes]
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
        eval_freq=BUDGET//(10*len(eval_mazes)), verbose=1,
        n_eval_episodes=len(eval_mazes),
        callback_after_eval=tb_callback,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=optimal_reward, verbose=1)
    )

    # model = PPO("MlpPolicy", env=train_env, ...)
    model = sb3_controller(
        PPO, policy="MlpPolicy", env=train_env, seed=SEED, learning_rate=1e-3)

    print("== Starting", "="*68)
    model.set_logger(configure(FOLDER, ["csv", "tensorboard"]))
    model.learn(BUDGET, callback=eval_callback, progress_bar=True)

    tb_callback.log_step(True)
    print("="*80)


def evaluate():
    model = load_sb3_controller(BEST)

    rng = random.Random(0)
    robot = Robot.BuildData.from_string("DD")

    n = 1000
    rewards = []

    print()
    print("="*80)
    print("Testing for generalization")
    _log_format = f"\r[{{:6.2f}}%] normalized reward: {{:.1g}} for {{}}"

    simulation = Simulation(Maze.from_string(""), robot)
    for i in range(n):
        maze_data = Maze.BuildData(
            width=rng.randint(10, 30),
            height=rng.randint(10, 30),
            seed=rng.randint(0, 10000),
            unicursive=False,
            clue=[Sign(value=1)],
            p_lure=0, p_trap=0
        )
        maze = Maze.generate(maze_data)
        simulation.reset(maze=maze)
        simulation.run(model)
        reward = simulation.normalized_reward()
        rewards.append(reward)
        print(_log_format.format(100*(i+1)/n, reward,
                                 maze_data.to_string()),
              end='', flush=True)
    print()

    avg_reward = sum(rewards) / n
    optimal = " (optimal)" if math.isclose(avg_reward, 1) else ""
    print(f"Average score of {avg_reward}{optimal} on {n} random mazes")
    print(f"> {100*sum(r for r in rewards if r == 1)/n}% solved")
    print("="*80)
    return avg_reward


def main():
    folder = pathlib.Path(FOLDER)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=False)

    train()
    evaluate()

    with CV2QTGuard(platform=False):
        amaze_main(f"--controller {BEST} --extension sb3 --maze {TRAIN_MAZE}"
                   f" --auto-quit")


if __name__ == "__main__":
    main()
