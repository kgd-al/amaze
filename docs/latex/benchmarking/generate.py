#!/bin/env python3

import os
import math
import time
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym

from pathlib import Path
from rich.progress import Progress

import amaze
from amaze.extensions.sb3.maze_env import MazeEnv


pd.options.display.float_format = '{:.3f}'.format

for v in [11, 15, 21]:
    for s in [5, 10, 20]:
        for io in ["DD", "CD", "CC"]:
            for l in [0, .5, 1]:
                gym.register(f"AMaze/AMaze.{io}{v}.{s}x{s}.{l}-v0",
                            lambda *args, _io=io: MazeEnv(maze=amaze.Maze.BuildData.from_string(f"M4_{s}x{s}_C1_l{l}_L.25"),
                                                        robot=amaze.Robot.BuildData.from_string(f"{_io}{v}"), *args))

registration = gym.envs.registration

def line(c='='):
    print(c*os.get_terminal_size()[0])

def short_name(item):
    return item.name.replace("Jax-", "").split("-")[0].split("_")[0].replace("Deterministic", "").replace("NoFrameskip", "")

def evaluate(env_name):
    env = gym.make(env_name)
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

#gym.pprint_registry()
registry = registration.registry
ordered_registry = {}
all_kwargs, kwargs_map = {}, {}
short_names, names = set(), {}
for name, item in registry.items():
    key = short_name(item)
    names[item.name] = key
    short_names.add(key)

    #pprint.pprint(item)
    version = item.version

    if item.namespace in ["dm_control"]:
        continue

    if "jax" in key.lower():
        continue

    family = item.namespace
    if family is None:
        for env_name, envs in [
            ("Box2D", ["BipedalWalker", "CarRacing", "LunarLander"]),
            ("Toy Text", ["Blackjack", "CliffWalking", "FrozenLake", "Taxi"]),
            ("Mujoco", ["Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup",
                        "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher",
                        "Swimmer", "Walker2d"]),
            ("Classic Control", ["Acrobot", "CartPole", "MountainCar", "Pendulum"]),
        ]:
            if any(n in key for n in envs):
                family = env_name
                #print(name, family, item.entry_point)
                break

        if family is None and isinstance(item.entry_point, str):
            family = item.entry_point.split(".")[-1].split(":")[0]

    if family in ["openai_gym_compatibility", "atari_env"]:
        continue

    #print(f"{key:>30s} v{version} {'('+item.name+')':^40s} from {family}")
    assert family is not None, f"Family not found: {pprint.pformat(item)}"
    assert family[0].upper() == family[0]

    if family not in ordered_registry:
        ordered_registry[family] = {}

    if key not in ordered_registry[family]:
        ordered_registry[family][key] = [0, set(), set(), []]

    kwargs = tuple(sorted(item.kwargs.keys()))

    if kwargs not in kwargs_map:
        kwargs_map[kwargs] = [len(kwargs_map), 0]
    kwargs_map[kwargs][1] += 1

    entry = ordered_registry[family][key]
    entry[0] += 1
    entry[1].add(kwargs_map[kwargs][0])
    entry[2].add(item.namespace)
    entry[3].append((version, name))

#pprint.pprint(ordered_registry)

candidates = {}
for family, entries in ordered_registry.items():
    f_candidates = []
    for short_name, data in sorted(entries.items()):
        #pprint.pprint(data)
        n, kwargs, namespaces, details = data
        assert len(namespaces) == 1
        namespace = next(iter(namespaces))
        if n > 1:
            v = registration.find_highest_version(namespace, short_name)
            candidate = f"{short_name}-v{v}"
            if namespace:
                candidate = f"{namespace}/{candidate}"
        else:
            candidate = details[0][1]
        f_candidates.append(candidate)

        for kw in kwargs:
            if family not in all_kwargs:
                all_kwargs[family] = {}

            _nkw = all_kwargs[family]
            if kw not in _nkw:
                _nkw[kw] = 0

            _nkw[kw] += 1

    candidates[family] = f_candidates

#pprint.pprint({k: v[0:3] for k,v in ordered_registry.items()})
#pprint.pprint(kwargs_map)
line()
print(f"{len(registry)} registered environments")
print("Candidate environments:")
for family, f_candidates in candidates.items():
    print(f"{len(f_candidates):3d} {family}")
line('-')
print("keyword arguments:")
pprint.pprint(all_kwargs)
pprint.pprint(kwargs_map)

line()
folder=Path(__file__).parent
datafile=folder.joinpath("table.csv")
print("Datafile:", datafile)

try:
    df = pd.read_csv(datafile, index_col="Name")
except FileNotFoundError:
    df = pd.DataFrame(columns=["Family", "Time", "Args"])
    df.index.name = "Name"

replicates = 10
with Progress() as progress:
    task = progress.add_task("[green] Timing", total=sum(len(fc) for fc in candidates.values()))

    for family, f_candidates in candidates.items():
        for candidate in f_candidates:
            progress.update(task, advance=1, description=candidate)

            if candidate not in df.index or math.isnan(df.loc[candidate]["Time"]):
                try:
                    start = time.time_ns()
                    for _ in range(replicates):
                        evaluate(candidate)
                    t = (time.time_ns() - start) / (replicates * 10**9)
                except Exception as e:
                    print(f"[ERROR] Processing {candidate}:\n{e}")
                    t = float("nan")

                if not math.isnan(t):
                    df.loc[candidate] = [family, t, len(gym.spec(candidate).kwargs)]
