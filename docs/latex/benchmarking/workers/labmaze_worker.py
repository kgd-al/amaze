from random import Random

import labmaze
from common import Progress, STEPS


def _evaluate(size):
    maze = labmaze.RandomMaze(height=size, width=size, random_seed=42)
    return True


def process(df):
    with Progress(df, "misc") as progress:
        progress.add_task(1)

        for size in [11]:
            candidate = f"labmaze-placeholder"
            progress.evaluate("LabMaze", candidate, _evaluate,
                              size=size)

        progress.close()

    df.loc["labmaze-placeholder", "Time"] = float("nan")

    return progress.errors
