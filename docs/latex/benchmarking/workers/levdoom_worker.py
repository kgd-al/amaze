import levdoom
from common import Progress, STEPS


def _evaluate(env_name):
    env = levdoom.make(env_name,
                       doom=dict(resolution="160X120", render=False))
    env.reset()

    for _ in range(STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()

    env.close()
    return True


def process(df):
    envs = [e for scenario in levdoom.env_mapping.values() for e in scenario]

    with Progress(df, "VizDoom") as progress:
        progress.add_task(len(envs))

        for env_name in envs:
            progress.evaluate("LevDoom", env_name, _evaluate, env_name=env_name)

        progress.close()

    return progress.errors
