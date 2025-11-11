import math
import itertools
import pprint

from common import Progress, STEPS

import dmc
from dmc_benchmark import TASKS


def _evaluate(task, obs):
    env = dmc.make(name=task, obs_type=obs,
                   frame_stack=3, action_repeat=1+(obs == "pixels"), seed=1)

    time = env.reset()
    for _ in range(STEPS):
        action = env.action_spec().generate_value()

        time = env.step(action)

        if time.last():
            env.reset()

    return True


def process(df):
    tasks = TASKS
    observations = ['states', 'pixels']
    envs = list(itertools.product(tasks, observations))

    with Progress(df, "misc") as progress:
        progress.add_task(len(envs))

        for task, obs in envs:
            progress.evaluate("URLB", f"{task}-{obs}", _evaluate, task=task, obs=obs)

        progress.close()

    return progress.errors
