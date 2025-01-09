#!/bin/env python3
import itertools
import math
import pprint
import signal
from pathlib import Path

import pandas as pd
from rich.progress import track

from amaze import Maze, Sign, InputType, Simulation
from amaze.simu import MazeMetrics
from amaze.simu.types import MazeClass

persistent_data = Path(__file__).parent.joinpath("data.csv")

try:
    df = pd.read_csv(persistent_data, index_col=0)
except FileNotFoundError:
    df = pd.DataFrame(columns=["ID", "Class", "Deceptiveness", "Surprisingness"])

seeds = range(200)
sizes = [5, 10, 20, 50]
signs = [Sign(value=v) for v in [.25, .5, .75, 1]]
clues = [[], [signs[3]], [signs[1], signs[3]], signs]
lures = [[], [signs[0]], [signs[0], signs[2]], signs]
traps = [[], [signs[1]], [signs[0], signs[2]], signs]
probas = [0, .25, .5, .75, 1]

items = [seeds, clues, lures, traps, probas, probas, sizes]
total = math.prod([len(i) for i in items])


def handler(*_):
    print("Panic-saving the dataframe")
    df.to_csv(persistent_data)
    exit(1)


for s in [signal.SIGINT, signal.SIGTERM]:
    signal.signal(signal.SIGINT, handler)


if len(df) < total:
    for seed, clue, lure, trap, p_lure, p_trap, size in track(
            itertools.product(*items), description="Processing", total=total):

        bd = Maze.BuildData(
            seed=seed,
            width=size, height=size,
            clue=clue,
            p_lure=p_lure, lure=lure,
            p_trap=p_trap, trap=trap,
            unicursive=(len(clue) == 0)
        )
        name = bd.to_string()
        if name not in df.index:
            maze = Maze.generate(bd)
            m_class = maze.maze_class()
            assert m_class is not MazeClass.INVALID, \
                f"{name} is invalid. From bd: {pprint.pformat(bd)}\n{pprint.pformat(maze.stats())}"

            metrics = Simulation.compute_metrics(maze, InputType.CONTINUOUS, 15)
            row = [name, m_class.name.capitalize(),
                   metrics[MazeMetrics.DECEPTIVENESS],
                   metrics[MazeMetrics.SURPRISINGNESS]["all"]]
            df.loc[name] = row
            print(len(df), *row)

    print(f"Processed {total} mazes, resulting in {len(df)} unique ones")
    df.to_csv(persistent_data)
