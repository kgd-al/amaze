import math
import pathlib
import random
import shutil
import time

from amaze.simu.controllers.tabular import TabularController
from amaze import Maze, Robot, Simulation, InputType, OutputType, StartLocation

ALPHA, GAMMA = 0.1, 0.5
FOLDER = pathlib.Path("tmp/demos/q_learning/")
ROBOT = Robot.BuildData(inputs=InputType.DISCRETE, outputs=OutputType.DISCRETE)


def train():
    start_time = time.time()

    maze_data = Maze.BuildData(width=20, height=20, seed=16, unicursive=True, p_lure=0, p_trap=0)
    print("Training with maze:", maze_data.to_string())
    train_mazes = [Maze.generate(maze_data.where(start=start)) for start in StartLocation]

    maze_data = maze_data.where(seed=14)
    print("Evaluating with maze:", maze_data.to_string())
    eval_mazes = [Maze.generate(maze_data.where(start=start)) for start in StartLocation]

    policy = TabularController(robot_data=ROBOT, epsilon=0.1, seed=0)
    simulation = Simulation(train_mazes[0], ROBOT)

    steps = [0, 0]

    n = 150
    _w = math.ceil(math.log10(n))
    _log_format = (
        f"\r[{{:6.2f}}%] Episode {{:{_w}d}}; train: {{:.2f}};" f" eval: {{:.2f}}; optimal: {{:.2f}}"
    )

    print()
    print("=" * 80)
    print("Training for a maximum of", n, "episodes")

    i = None
    for i in range(n):
        simulation.reset(train_mazes[i % len(train_mazes)])
        t_reward = q_train(simulation, policy)
        steps[0] += simulation.timestep

        policy.epsilon = 0.1 * (1 - i / n)

        e_rewards, en_rewards = [], []
        for em in eval_mazes:
            simulation.reset(em)
            e_rewards.append(q_eval(simulation, policy))
            en_rewards.append(simulation.infos()["pretty_reward"])
            steps[1] += simulation.timestep
        e_rewards = sum(e_rewards) / len(e_rewards)
        en_rewards = sum(en_rewards) / len(en_rewards)

        print(
            _log_format.format(100 * (i + 1) / n, i, t_reward, e_rewards, en_rewards),
            end="",
            flush=True,
        )

        if math.isclose(en_rewards, 1):
            print()
            print("[!!!!!!!] Optimal policy found [!!!!!!!]")
            break
        elif i == n - 1:
            print()

    print(
        f"Training took {time.time() - start_time:.2g} seconds for:\n"
        f" > {i} episodes\n"
        f" > {steps[0]} training steps\n"
        f" > {steps[1]} evaluating steps"
    )

    return policy


def q_train(simulation, policy):
    state = simulation.generate_inputs().copy()
    action = policy(state)

    while not simulation.done():
        reward = simulation.step(action)
        state_ = simulation.observations.copy()
        action_ = policy(state)
        policy.q_learning(state, action, reward, state_, action_, alpha=ALPHA, gamma=GAMMA)
        state, action = state_, action_

    return simulation.robot.reward


def q_eval(simulation, policy):
    action = policy.greedy_action(simulation.observations)
    while not simulation.done():
        simulation.step(action)
        action = policy.greedy_action(simulation.observations)

    return simulation.robot.reward


def evaluate_generalization(policy):
    policy.epsilon = 0
    rng = random.Random(0)

    n = 1000
    rewards = []

    print()
    print("=" * 80)
    print("Testing for generalization")

    print("\n-- Navigation", "-" * 66)
    _log_format = "\r[{:6.2f}%] normalized reward: {:.1g} for {}"

    for i in range(n):
        maze_data = Maze.BuildData(
            width=rng.randint(10, 30),
            height=rng.randint(10, 20),
            seed=rng.randint(0, 10000),
            unicursive=True,
            start=rng.choice([sl for sl in StartLocation]),
            p_lure=0,
            p_trap=0,
        )
        maze = Maze.generate(maze_data)
        simulation = Simulation(maze, ROBOT)
        simulation.run(policy)
        reward = simulation.normalized_reward()
        rewards.append(reward)
        print(
            _log_format.format(100 * (i + 1) / n, reward, maze_data.to_string()),
            end="",
            flush=True,
        )
    print()

    avg_reward = sum(rewards) / n
    optimal = " (optimal)" if math.isclose(avg_reward, 1) else ""
    print(f"Average score of {avg_reward}{optimal} on {n} random mazes")

    print("\n-- Inputs", "-" * 70)
    print(Simulation.inputs_evaluation(FOLDER, policy, signs=dict()))

    print("=" * 80)


def main(is_test=False):
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    FOLDER.mkdir(parents=True, exist_ok=False)

    policy = train()

    policy_file = policy.save(FOLDER.joinpath("policy"), dict(comment="Can solve unicursive mazes"))
    print("Saved optimized policy to", policy_file)

    evaluate_generalization(policy)


if __name__ == "__main__":
    main()
