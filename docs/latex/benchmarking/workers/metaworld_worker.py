import random
import metaworld
from common import Progress, STEPS


def _evaluate(env_class):
    env = env_class()
    env.reset()

    for _ in range(STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()
    return True


def process(df):
    benchmarks = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.items()

    with Progress(df, "misc") as progress:
        progress.add_task(len(benchmarks))

        for env_name, env_class in benchmarks:
            progress.evaluate("Metaworld", env_name, _evaluate, env_class=env_class)

        progress.close()

    return progress.errors
