import pprint

from common import line, Progress, STEPS

import retro

# exit(42)

def _short_name(item):
    return item.name.replace("Jax-", "").split("-")[0].split("_")[0].replace("Deterministic", "").replace("NoFrameskip", "")


def _evaluate(env_name):
    try:
        env = retro.make(game=env_name)
    except FileNotFoundError:
        return False

    observation = env.reset()

    for _ in range(STEPS):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()

    env.close()
    return True


def process(df):
    line("-")

    candidates = retro.data.list_games()
    with Progress(df, "Retro-Gym") as progress:
        progress.add_task(len(candidates))

        for candidate in candidates:
            family = candidate.split("-")[-1]
            progress.evaluate(family, candidate, _evaluate, env_name=candidate)

        progress.close()

    return progress.errors
