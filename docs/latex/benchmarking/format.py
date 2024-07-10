#!/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

import amaze

from pathlib import Path

import pprint

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

folder=Path(__file__).parent
fname = "table" + ("-detailed" if detailed else "")
datafile = folder.joinpath(f"{fname}.csv")

df = pd.read_csv(datafile, index_col="Name")

section("Raw data")
print(df)
df.to_csv(datafile)

section("Grouped data")
if detailed:
    group_keys = ["Library", "Family"]
    sort_keys = ["Library", ("Time", "median")]
    for amaze_char, amaze_type in zip("DHC", ["Discrete", "Hybrid", "Continuous"]):
        df.replace("AMaze-" + amaze_char, amaze_type, inplace=True)
else:
    group_keys = "Family"
    sort_keys = ("Time", "median")
gb = df.groupby(group_keys)
table = gb.agg({"Time": ['min', 'max', 'median', 'mean', 'std'], "Args": ["max"]})
table["N"] = gb.count()["Time"]
table.insert(0, "N", table.pop("N"))
table.sort_values(sort_keys, axis="rows", inplace=True)
print(table)

section("Labeled data")
def _key(library, family): return (library, family) if detailed else family
manual_data = {
    _key("Gymnasium", "Classic Control"): ["Continuous", "Both", "-"],
    _key("Gymnasium", "Toy Text"): ["Discrete", "Discrete", "-"],
    _key("Gymnasium", "Box2D"): ["Continuous", "Both", "-"],
    _key("Gymnasium", "Mujoco"): ["Continuous", "Continuous", "-"],
    _key("Gymnasium", "ALE"): ["Image", "Discrete", "Modes"],
}
if detailed:
    manual_data.update({
        _key("AMaze", "Discrete"): ["Discrete", "Discrete", "Extensive"],
        _key("AMaze", "Hybrid"): ["Image", "Discrete", "Extensive"],
        _key("AMaze", "Continuous"): ["Continuous", "Continuous", "Extensive"],
        _key("Procgen", "Procgen"): ["Image", "Discrete", "Modes"],
    })
else:
    manual_data.update({
        "AMaze": ["Both", "Both", "Extensive"]
    })

md_table = table[[("N", ""), ("Time", "median")]]
#md_table.columns = ["N", "Median time (s)"]

for c in ["Control", "Outputs", "Inputs"]:
    md_table.insert(1, c, ["?" for _ in range(len(md_table))])

for key, (inputs, outputs, control) in manual_data.items():
    md_table.loc[key, "Inputs"] = inputs
    md_table.loc[key, "Outputs"] = outputs
    md_table.loc[key, "Control"] = control

print(md_table)
md_table.to_markdown(folder.joinpath(f"gym_{fname}.md"), floatfmt=".3f", tablefmt="grid")
md_table.to_latex(folder.joinpath(f"gym_{fname}.tex"), float_format="%.3f")

table_pdf = folder.joinpath(f"gym_{fname}.pdf")

keys = sorted(gb.groups.keys())
hlines = []
if detailed:
    lib_order = {k: i for i, k in enumerate(sorted(set(t[0] for t in keys)))}
    positions = [-md_table.index.get_loc(f)-.5*lib_order[f[0]] for f in keys]

    hlines = [.25-sum(1 for _k in keys if _k[0] == k) - .75*i for i, k in enumerate(list(lib_order.keys())[:-1])]
    hlines = [hlines[i] if i == 0 else hlines[i] + hlines[i-1] for i in range(len(hlines))]
    #print(lib_order)
    #print(hlines)

else:
    lib_order = {}
    positions = [-md_table.index.get_loc(f) for f in keys]

# Cancel this one to get something readable
#df.loc["CarRacing-v2"] = float("nan")

df.boxplot(column="Time", by="Family", vert=False, positions=positions)
plt.savefig(folder.joinpath(f"gym_{fname}.png"))

ax = df.boxplot(column="Time", by=["Library", "Family"], vert=False, positions=positions, figsize=(3, .35*len(positions)),
                flierprops=dict(markerfacecolor='black', markeredgecolor='gray', marker='.', markersize=2))
plt.suptitle("")
ax.set_title("")
#ax.xlabel("Time (s)")

if True:
    ax.set_yticks([])
    #ax.set_yticklabels([])

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
    for y, ay, dy, t, va in [(-1.5, -1.4, 1.5, "Faster &\nNo control", "bottom"),
                            (-2.6, -2.6, -2.5, "Slower &\nLow control", "top")]:
        ax.text(5.5, y, t, ha="right", va=va, size="x-small", color="red")
        plt.arrow(7, ay, 0, dy, color="red", head_width=1, head_length=.5, length_includes_head=True, clip_on=False)
    ax.set_xlim(right=6)

for side in ['left', 'right', 'bottom']:
    ax.spines[side].set_visible(False)
plt.savefig(table_pdf, bbox_inches='tight', pad_inches=0.015)

pretty_tex = folder.joinpath(f"gym_pretty_{fname}.tex")
with open(pretty_tex, "w") as f:
    lc = md_table.columns[-1]
    table_str = md_table.to_latex(float_format="%.3f")
    tr = table_str.split("\n")
    print("-"*80)
    print("\n".join(tr))

    tr[0] = tr[0][:-2] + "c@{ }r@{}}"
    tr[2] = "&".join((r"\multirow{2}{*}{" + h + r"}" if 0 < i < (5+detailed) else h) for i, h in enumerate(tr[2].split("&")))
    tr[2] = tr[2].replace(r"Time \\", r"\multicolumn{2}{c}{Time (s)} \\ ")
    tr.insert(3, fr"\cmidrule(lr){{{6+detailed}-{7+detailed}}}")

    multirow = 7.77 if not detailed else 11.6

    tr[4] = "&".join(
        tr[5].split("&")[0:1+detailed]
        + tr[4].split("&")[1+detailed:-1]
        + [
            r" Median & \multirow{" + str(multirow) + "}{*}{"
            + r"\includegraphics[height=\img]{"
            + str(table_pdf) + r"}} \\"
        ])
    del tr[5]
    tr[5] = fr"\cmidrule(r){{1-{6+detailed}}}"
    #tr[5] = fr"\cmidrule(r){{1-{7+detailed}}}"

    if detailed:
        for i in [9, 15]:
            tr[i] = tr[i].replace("cline", "cmidrule(r)")
        #tr[9] = r"\cmidrule(r){1-8}"
        del tr[-4]
        #del tr[-2]
        tr[6] = " & ".join([r"\textbf{AMaze}"] + tr[6].split(" & ")[1:])
        img_height = 11.25

    else:
        tr[8] = " & ".join(r"\textbf{" + v + r"}" for v in tr[8].replace(r"\\", "").split(" & ")) + r"\\"
        img_height = 7.2

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
