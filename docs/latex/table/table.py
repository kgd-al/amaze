#!/bin/env python3

import os
import math
import time
import pprint
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

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
line()
print(df)
df.to_csv(datafile)

line()

gb = df.groupby("Family")
table = gb.agg({"Time": ['min', 'max', 'median', 'mean', 'std'], "Args": ["max"]})
table["N"] = gb.count()["Time"]
table.insert(0, "N", table.pop("N"))
table.sort_values(("Time", "median"), axis="rows", inplace=True)
print(table)

line()

manual_data = {
    "Classic Control": ["Continuous", "Both", "Random init"],
    "Toy Text": ["Discrete", "Discrete", "Random init"],
    "Box2D": ["Continuous", "Both", "Random init"],
    "Mujoco": ["Continuous", "Continuous", "Random init"],
    "ALE": ["Image", "Discrete", "Modes"],
    "AMaze": ["Both", "Both", "Extensive"]
}

md_table = table[[("N", ""), ("Time", "median")]]
md_table.columns = ["N", "Median time (s)"]

for c in ["Control", "Outputs", "Inputs"]:
    md_table.insert(1, c, ["?" for _ in range(len(md_table))])

for family, (inputs, outputs, control) in manual_data.items():
    md_table.loc[family, "Inputs"] = inputs
    md_table.loc[family, "Outputs"] = outputs
    md_table.loc[family, "Control"] = control

md_table.to_markdown(folder.joinpath("gym_table.md"), floatfmt=".3f", tablefmt="grid")
md_table.to_latex(folder.joinpath("gym_table.tex"), float_format="%.3f")

table_pdf = folder.joinpath("gym_table.pdf")
with amaze.extensions.sb3.CV2QTGuard():
    positions = [-md_table.index.get_loc(f) for f in sorted(df["Family"].unique())]

    # Cancel this one to get something readable
    #df.loc["CarRacing-v2"] = float("nan")

    df.boxplot(column="Time", by="Family", vert=False, positions=positions)
    plt.savefig(folder.joinpath("gym_table.png"))

    df.boxplot(column="Time", by="Family", vert=False, positions=positions, figsize=(3, 2))
    plt.title("")
    plt.suptitle("")
    plt.xlabel("Time (s)")
    #plt.yticks([])
    plt.ylabel("")
    plt.xscale("log")
    plt.gca().tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.gca().xaxis.set_label_position('top')
    plt.savefig(table_pdf, bbox_inches='tight')

with open(folder.joinpath("gym_pretty_table.tex"), "w") as f:
    lc = md_table.columns[-1]
    table_str = md_table.to_latex(float_format="%.3f")
    table_str = table_str.replace(r"lr}", r"lrr}")
    table_str = table_str.replace(lc, f"\\multicolumn{{2}}{{c}}{{{lc}}}")

    def _print(*args): print(*args, file=f)
    _print(r"\documentclass{standalone}")
    _print(r"\usepackage{booktabs}")
    _print(r"\usepackage{tikz}")
    _print(r"\usetikzlibrary{calc, positioning}")
    _print(r"\begin{document}")
    _print(r"\begin{tikzpicture}")
    _print(r" \node (T){")
    _print(table_str)
    _print(r" };")
    _print(r" \path let \p{A} = ($(T.north)-(T.south)$) in")
    _print(r"  node (I) [above right=0 of T.south east, scale=.8, xshift=-2cm]")
    _print(r"   {\includegraphics[height=\y{A}]{", table_pdf, r"}};")
    _print(r"\end{tikzpicture}")
    _print(r"\end{document}")

    #_print(r"\documentclass{standalone}")
    #_print(r"\usepackage{booktabs}")
    #_print(r"\usepackage{tikz}")
    #_print(r"\usetikzlibrary{calc, positioning}")
    #_print(r"\begin{document}")
    #_print(r"\begin{tikzpicture}")
    #_print(r" \node (T){")
    #_print(md_table.to_latex(float_format="%.3f"))
    #_print(r" };")
    #_print(r" \path let \p{A} = ($(T.north)-(T.south)$) in")
    #_print(r"  node (I) [above right=0 of T.south east, scale=.8, yshift=-1cm]")
    #_print(r"   {\includegraphics[height=\y{A}]{", table_pdf, r"}};")
    #_print(r"\end{tikzpicture}")
    #_print(r"\end{document}")
