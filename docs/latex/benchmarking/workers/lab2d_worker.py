from random import Random

import dmlab2d
import dmlab2d.runfiles_helper
import numpy as np
from common import Progress, STEPS


def _evaluate(env_name, observation):
    seed = 0

    lab2d = dmlab2d.Lab2d(dmlab2d.runfiles_helper.find(),
                          dict(levelName=env_name))
    env = dmlab2d.Environment(lab2d, [observation], seed=seed)

    rng = Random(seed)
    actions = []
    for name, spec in env.action_spec().items():
        if spec.dtype == np.dtype('int32'):
            actions.append((name, lambda: rng.randint(spec.minimum, spec.maximum)))
        elif spec.dtype == np.dtype('float64'):
            actions.append((name, lambda: rng.uniform(spec.minimum, spec.maximum)))

    timestep = env.reset()
    for _ in range(STEPS):
        action = {name: gen() for name, gen in actions}
        timestep = env.step(action)
        if timestep.last():
            env.reset()

    env.close()
    return True


def process(df):
    steps, text, layer, rgb = "WORLD.STEPS", "WORLD.TEXT", "WORLD.LAYER", "WORLD.RGB"
    envs = {
        'chase_eat': [steps, text, layer, rgb],
        'clean_up': [rgb],
        'commons_harvest': [rgb],
        'pushbox': [steps, text, layer, rgb],
        'running_with_scissors': [rgb],
    }
    envs = [(name, obs) for name, observations in envs.items() for obs in observations]

    with Progress(df, "misc") as progress:
        progress.add_task(len(envs))

        for env_name, obs in envs:
            candidate = f"{env_name}-{obs.split('.')[1].lower()}"
            progress.evaluate("Lab2D", candidate, _evaluate,
                              env_name=env_name, observation=obs)

        progress.close()

    return progress.errors
