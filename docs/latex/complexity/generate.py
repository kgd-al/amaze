#!/bin/env python3

import pprint
import signal
from pathlib import Path
from random import Random

import pandas as pd
from rich.progress import Progress

from amaze import Maze, Sign, InputType, Simulation
from amaze.simu import MazeMetrics
from amaze.simu.types import MazeClass

persistent_data = Path(__file__).parent.joinpath("data.csv")

try:
    # raise FileNotFoundError()
    df = pd.read_csv(persistent_data, index_col=0)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Class", "Deceptiveness", "Surprisingness"])
    df.index.name = "ID"

n = 100000


def build_data(_rng: Random, _m_class: MazeClass):
    def _signs(): return [Sign(value=.5 * _rng.random() + .5) for _ in range(_rng.randint(2, 4))]
    _size = _rng.randint(2, 50)
    _clues = _signs() if _m_class.value & 0b001 else []
    _p_lure, _lures = 0, []
    _p_trap, _traps = 0, []
    if (_m_class.value & 0b010) or _m_class.value is MazeClass.TRIVIAL:
        _p_lure = .75 * _rng.random() + .25
        _lures = _signs()
    if _m_class.value & 0b100:
        _p_trap = .75 * _rng.random() + .25
        _traps = _signs()
    return Maze.BuildData(
        seed=_rng.randint(0, 2 ** 32),
        width=_size, height=_size,
        clue=_clues,
        p_lure=_p_lure, lure=_lures,
        p_trap=_p_trap, trap=_traps,
        unicursive=_m_class is MazeClass.TRIVIAL,
    )


def handler(*_):
    print("Panic-saving the dataframe")
    df.to_csv(persistent_data)
    exit(1)


for s in [signal.SIGINT, signal.SIGTERM]:
    signal.signal(signal.SIGINT, handler)

nmc = len(MazeClass) - 1
total = nmc * n
if len(df) < total:
    queried, generated, skipped, kept = 0, 0, 0, 0
    sizes = {c: 0 for c in MazeClass if c is not MazeClass.INVALID}
    m_classes = list(sizes.keys())

    rng = Random(0)

    with Progress() as progress:
        t_task = progress.add_task(description="Total", total=total)
        progress.advance(t_task, len(df))

        while len(df) < total:
            m_class = m_classes[len(df) % nmc]
            pm_class = m_class.name.capitalize()

            bd = build_data(rng, m_class)
            name = bd.to_string()
            queried += 1

            # print(name, (seed, clue, lure, trap, p_lure, p_trap, size))
            if name not in df.index:
                maze = Maze.generate(bd)
                generated += 1

                derived_class = maze.maze_class()
                if m_class is not derived_class:
                    print("Skipped", name)
                    skipped += 1
                    continue
                assert m_class is not MazeClass.INVALID, \
                    f"{name} is invalid. From bd: {pprint.pformat(bd)}\n{pprint.pformat(maze.stats())}"

                metrics = Simulation.compute_metrics(maze, InputType.CONTINUOUS, 15)
                row = [pm_class,
                       metrics[MazeMetrics.DECEPTIVENESS],
                       metrics[MazeMetrics.SURPRISINGNESS]["all"]]
                df.loc[name] = row
                print(len(df), kept, name, *row)
                # print()

                progress.advance(t_task)
                kept += 1
                sizes[m_class] += 1

    print(f"Processed {generated} mazes ({queried} queried), dropping {skipped} and keeping {kept}")
    df.to_csv(persistent_data)

else:
    print("Filesystem data has expected size. Reusing as-is")

print(df)
