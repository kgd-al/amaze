import math
import pathlib
import random
import shutil
import time

from amaze.simu.controllers.control import controller_factory, save, check_types
from amaze.simu.controllers.tabular import TabularController
from amaze.simu.maze import Maze, StartLocation
from amaze.simu.robot import Robot
from amaze.simu.simulation import Simulation
from amaze.simu.types import InputType, OutputType

ALPHA = 0.1
GAMMA = 0.5

FOLDER = pathlib.Path("tmp/demos/q_learning/")


def robot_build_data():
    return Robot.BuildData(
        inputs=InputType.DISCRETE,
        outputs=OutputType.DISCRETE,
        control="tabular",
        control_data=dict(
            actions=Simulation.discrete_actions(),
            epsilon=0.1, seed=0
        )
    )


def train():
    start_time = time.time()

    maze_data = Maze.BuildData(
        width=20, height=20,
        seed=16,
        unicursive=True,
        p_lure=0, p_trap=0
    )
    print("Training with maze:", maze_data.to_string())
    train_mazes = [
        Maze.generate(maze_data.where(start=start))
        for start in StartLocation
    ]

    maze_data = maze_data.where(seed=14)
    print("Evaluating with maze:", maze_data.to_string())
    eval_mazes = [
        Maze.generate(maze_data.where(start=start))
        for start in StartLocation
    ]

    robot = robot_build_data()
    policy: TabularController = controller_factory(robot.control,
                                                   robot.control_data)
    assert check_types(policy, robot)

    simulation = Simulation(train_mazes[0], robot)

    steps = [0, 0]

    n = 150
    _w = math.ceil(math.log10(n))
    _log_format = (f"\r[{{:6.2f}}%] Episode {{:{_w}d}}; train: {{:.2f}};"
                   f" eval: {{:.2f}}; optimal: {{:.2f}}")

    print()
    print("="*80)
    print("Training for a maximum of", n, "episodes")

    i = None
    for i in range(n):
        simulation.reset(train_mazes[i % len(train_mazes)])
        t_reward = q_train(simulation, policy)
        steps[0] += simulation.timestep

        policy.epsilon = .1 * (1 - i / n)

        e_rewards, en_rewards = [], []
        for em in eval_mazes:
            simulation.reset(em)
            e_rewards.append(q_eval(simulation, policy))
            en_rewards.append(simulation.infos()["pretty_reward"])
            steps[1] += simulation.timestep
        e_rewards = sum(e_rewards) / len(e_rewards)
        en_rewards = sum(en_rewards) / len(en_rewards)

        print(_log_format.format(100*(i+1)/n, i,
                                 t_reward, e_rewards, en_rewards),
              end='', flush=True)

        if math.isclose(en_rewards, 1):
            print()
            print("[!!!!!!!] Optimal policy found [!!!!!!!]")
            break
        elif i == n-1:
            print()

    print(f"Training took {time.time() - start_time:.2g} seconds for:\n"
          f" > {i} episodes\n"
          f" > {steps[0]} training steps\n"
          f" > {steps[1]} evaluating steps")

    return policy


def q_train(simulation, policy):
    state = simulation.generate_inputs().copy()
    action = policy(state)

    while not simulation.done():
        reward = simulation.step(action)
        state_ = simulation.observations.copy()
        action_ = policy(state)
        policy.q_learning(state, action, reward, state_, action_,
                          alpha=ALPHA, gamma=GAMMA)
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
    robot = robot_build_data()

    n = 1000
    rewards = []

    print()
    print("="*80)
    print("Testing for generalization")
    _log_format = f"\r[{{:6.2f}}%] normalized reward: {{:.1g}} for {{}}"

    for i in range(n):
        maze_data = Maze.BuildData(
            width=rng.randint(10, 30),
            height=rng.randint(10, 20),
            seed=rng.randint(0, 10000),
            unicursive=True,
            p_lure=0, p_trap=0
        )
        maze = Maze.generate(maze_data)
        simulation = Simulation(maze, robot)
        simulation.run(policy)
        reward = simulation.normalized_reward()
        rewards.append(reward)
        print(_log_format.format(100*(i+1)/n, reward,
                                 maze_data.to_string()),
              end='', flush=True)
    print()

    avg_reward = sum(rewards) / n
    optimal = " (optimal)" if math.isclose(avg_reward, 1) else ""
    print(f"Average score of {avg_reward}{optimal} on {n} random mazes")
    print("="*80)


def main():
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    FOLDER.mkdir(parents=True, exist_ok=False)

    policy = train()

    policy_file = save(policy, FOLDER.joinpath("policy"),
                       dict(comment="Can solve unicursive mazes"))
    print("Saved optimized policy to", policy_file)

    evaluate_generalization(policy)


if __name__ == "__main__":
    main()
