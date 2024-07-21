import itertools

import mazeexplorer
from common import Progress, STEPS


def _evaluate(size, resolution, complexity, density):
    maze = mazeexplorer.MazeExplorer(
        number_maps=1,
        random_spawn=True, random_textures=True, random_key_positions=True,
        seed=0,
        size=(size, size), scaled_resolution=(resolution, resolution),
        complexity=complexity, density=density)
    maze.reset()
    for _ in range(STEPS):
        obs, rewards, dones, info = maze.step(maze.action_space.sample())
        if dones:
            maze.reset()


def process(df):
    args = dict(
        size=[10, 20, 50],
        resolution=[11, 21, 42],
        complexity=[0, .5, 1],
        density=[0, .5, 1],
    )
    args = [dict(zip(args.keys(), t)) for t in itertools.product(*args.values())]

    with Progress(df, "VizDoom") as progress:
        progress.add_task(len(args))

        for kwargs in args:
            candidate = "-".join([f"{k[0]}{v}" for k, v in sorted(kwargs.items())])
            progress.evaluate("MazeExplorer", candidate, _evaluate,
                              **kwargs)

        progress.close()

    return progress.errors
