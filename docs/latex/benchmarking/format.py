#!/bin/env python3

import os
import pprint
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

detailed = (len(sys.argv) > 1)


def line(msg=None, c='='):
    n = os.get_terminal_size()[0]
    if msg is not None:
        n -= len(msg) + 4
        print(2*c, msg, end=' ')
    print(c*n)


def section(name):
    print()
    line()
    line(name)


folder = Path(__file__).parent
fname = "table" + ("-detailed" if detailed else "")
datafile = folder.joinpath(f"table.csv")

df = pd.read_csv(datafile, index_col="Name")
if detailed:
    df["Family"] += df["Class"].fillna("")
else:
    df = df[(df.Library == "AMaze") | (df.Library == "Gymnasium")
            | (df.Family == "MazeExplorer") | (df.Family == "Lab2D")]

misc = "Miscellaneous"
df.loc[(df.Library == "misc"), "Library"] = misc

section("Raw data")
df = df[df.Family != "LabMaze"]
# for family in ["LabMaze", "ObstacleTower"]:
#     df.loc[f"{family.lower()}-placeholder",:] = [misc, family, "", float("nan")]
print(df)

section("Grouped data")
if detailed:
    group_keys = ["Library", "Family"]
    sort_keys = ["Library", ("Time", "median")]
    for amaze_char, amaze_type in zip("DHC", ["Discrete", "Hybrid", "Continuous"]):
        df.replace("AMaze" + amaze_char, amaze_type, inplace=True)
else:
    group_keys = "Family"
    sort_keys = ("Time", "median")
gb = df.groupby(group_keys)
table = gb.agg({"Time": ['min', 'max', 'median', 'mean', 'std']})
table["N"] = gb.count()["Time"]
table.insert(0, "N", table.pop("N"))
table["N"] = table["N"].astype("Int64")
table.sort_values(sort_keys, axis="rows", inplace=True)
print(table)

section("Labeled data")
no_control = "None"
def _key(library, family): return (library, family) if detailed else family
manual_data = {
    _key("Gymnasium", "Classic Control"): ["Continuous", "Both", no_control],
    _key("Gymnasium", "Toy Text"): ["Discrete", "Discrete", no_control],
    _key("Gymnasium", "Box2D"): ["Continuous", "Both", no_control],
    _key("Gymnasium", "Mujoco"): ["Continuous", "Continuous", no_control],
    _key("Gymnasium", "ALE"): ["Image", "Discrete", "Modes"],
    _key("VizDoom", "MazeExplorer"): ["Image", "Discrete", "Extensive"],
    _key(misc, "Lab2D"): ["Both", "Discrete", "Script"],
}
if detailed:
    manual_data.update({
        _key("AMaze", "Discrete"): ["Discrete", "Discrete", "Extensive"],
        _key("AMaze", "Hybrid"): ["Image", "Discrete", "Extensive"],
        _key("AMaze", "Continuous"): ["Continuous", "Continuous", "Extensive"],
        _key("VizDoom", "LevDoom"): ["Image", "Discrete", no_control],
        # _key(misc, "LabMaze"): ["Discrete", "Discrete", "Intermediate"],
        _key(misc, "Metaworld"): ["Continuous", "Continuous", no_control],
        _key(misc, "ProcGen"): ["Image", "Discrete", "Modes"],
        # _key(misc, "ObstacleTower"): ["Image", "Discrete", "Extensive"],
        _key(misc, "URLB"): ["Both", "Continuous", "None"],
        _key("Retro-Gym", "GameBoy"): [
            r"\multirow{5}{*}{Image}",
            r"\multirow{5}{*}{Discrete}",
            r"\multirow{5}{*}{None}",
        ]
    })
    manual_data.update({
        _key("Retro-Gym", console): ["", "", ""]
        for console in ["Sms", "Nes", "Genesis", "Snes"]
    })
else:
    manual_data.update({
        "AMaze": ["Both", "Both", "Extensive"]
    })

md_table = table[[("N", ""), ("Time", "median")]]

for c in ["Control", "Outputs", "Inputs"]:
    md_table.insert(1, c, ["?" for _ in range(len(md_table))])

for key, (inputs, outputs, control) in manual_data.items():
    md_table.loc[key, "Inputs"] = inputs
    md_table.loc[key, "Outputs"] = outputs
    md_table.loc[key, "Control"] = control

print(md_table)
md_table.to_latex(folder.joinpath(f"gym_{fname}.tex"), float_format="%.3f")

table_pdf = folder.joinpath(f"gym_{fname}.pdf")

hlines = []
if detailed:
    keys = gb.groups.keys()
    lib_order = {k: i for i, k in enumerate(sorted(set(t[0] for t in keys)))}

    positions = [-md_table.index.get_loc(f) - .5 * lib_order.get(f[0], 0) for f in keys]

    hlines = [.25-sum(1 for _k in keys if _k[0] == k) - .75*(i>0) for i, k in enumerate(list(lib_order.keys())[:-1])]
    hlines = [sum(hlines[:i+1]) for i in range(len(hlines))]

    scale = .8
    positions = [scale * p for p in positions]
    hlines = [scale * h for h in hlines]
    # print(hlines)

else:
    keys = sorted(df["Family"].unique())
    print(keys)
    lib_order = {}
    positions = [-md_table.index.get_loc(f) for f in keys]
    print("__")
    print(positions)
    print("__")

pprint.pprint(dict(zip(keys, positions)))

df.boxplot(column="Time", by="Family", vert=False, positions=positions)
plt.savefig(folder.joinpath(f"gym_{fname}.png"))

ax = df.boxplot(column="Time", by=group_keys, vert=False,
                positions=positions, figsize=(3, .35*len(positions)),
                flierprops=dict(markerfacecolor='black', markeredgecolor='gray', marker='.', markersize=2))
plt.suptitle("")
ax.set_title("")
#ax.xlabel("Time (s)")

if True:
    ax.set_yticks([])
    # ax.set_yticklabels([])

ax.set_ylabel("")
ax.set_xscale("log")
ax.tick_params(which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
ax.xaxis.set_label_position('top')

if detailed:
    ax.set_ylim(top=.6)
    for y in hlines:
        ax.axhline(y=y, linewidth=.6, color="black")

    for _, y in md_table.loc[("AMaze",), "Time"].iterrows():
        ax.axvline(x=y["median"], linestyle="dashed", linewidth=.3, color="green")

else:
    ax.set_ylim(top=.75)
    for y in [1, 2]:
        ax.axhline(y=-1*y-.5, linestyle="dashed", linewidth=.5, color="red")
    for y, ay, dy, t, va in [(-1.5, -1.4, 2.0, "Faster &\nNo control", "bottom"),
                             (-2.6, -2.6, -4.5, "Slower &\nLow control", "top")]:
        ax.text(5.5, y, t, ha="right", va=va, size="x-small", color="red")
        plt.arrow(7, ay, 0, dy, color="red", head_width=1, head_length=.5, length_includes_head=True, clip_on=False)
    ax.set_xlim(right=6)

for side in ['left', 'right', 'bottom']:
    ax.spines[side].set_visible(False)
plt.savefig(table_pdf, bbox_inches='tight', pad_inches=0.015)

pretty_tex = folder.joinpath(f"gym_pretty_{fname}.tex")
cols, overdraw = 6 + detailed, False
with open(pretty_tex, "w") as f:
    lc = md_table.columns[-1]
    table_str = (md_table
                 .to_latex(float_format="%.3f")
                 .replace(fr"\cline{{1-{cols}}}", fr"\cmidrule(r){{1-{cols+overdraw}}}")
                 .replace("NaN", "-"))
    tr = table_str.split("\n")
    print("-"*80)
    print("\n".join(tr))

    tr[0] = tr[0][:-2] + "c@{ }r@{}}"
    tr[2] = "&".join((r"\multirow{2}{*}{" + h + r"}" if 0 < i < (5+detailed) else h) for i, h in enumerate(tr[2].split("&")))
    tr[2] = tr[2].replace(r"Time \\", r"\multicolumn{2}{c}{Time (s)} \\ ")
    tr.insert(3, fr"\cmidrule(lr){{{6+detailed}-{7+detailed}}}")

    multirow = 9.77 if not detailed else 22.53

    tr[4] = "&".join(
        tr[5].split("&")[0:1+detailed]
        + tr[4].split("&")[1+detailed:-1]
        + [
            " Median &\n" + r" \multirow{" + str(multirow) + "}{*}{"
            + r"\includegraphics[height=\img]{"
            + str(table_pdf) + r"}} \\"
        ])
    del tr[5]

    if detailed:
        del tr[-4]
        #del tr[-2]
        tr[6] = " & ".join([r"\textbf{AMaze}"] + tr[6].split(" & ")[1:])

        img_height = 22.125

    else:
        tr[8] = " & ".join(r"\textbf{" + v + r"}" for v in tr[8].replace(r"\\", "").split(" & ")) + r"\\"
        img_height = 9.2

    print("-"*80)
    print("\n".join(tr))
    print("-"*80)


    def _print(*args): print(*args, file=f)
    _print(r"\documentclass{standalone}")
    _print(r"\usepackage{booktabs}")
    _print(r"\usepackage{multirow}")
    _print(r"\usepackage{tikz}")
    _print(r"\usetikzlibrary{calc, positioning, tikzmark}")
    _print(r"\begin{document}")
    _print(r"\newlength{\img}%")
    _print(fr"\setlength{{\img}}{{{img_height}\baselineskip}}%") #7.5
    _print("\n".join(tr))
    _print(r"\end{document}")

os.chdir(folder)
os.system(f"pdflatex -halt-on-error -interaction=batchmode {pretty_tex}")
print(f"Generated {pretty_tex}: {pretty_tex.exists()}")
