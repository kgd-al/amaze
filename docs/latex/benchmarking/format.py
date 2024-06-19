import os
import pandas as pd
import matplotlib.pyplot as plt

import amaze

from pathlib import Path

def line(c='='):
    print(c*os.get_terminal_size()[0])

folder=Path(__file__).parent
datafile=folder.joinpath("table.csv")

df = pd.read_csv(datafile, index_col="Name")

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
#md_table.columns = ["N", "Median time (s)"]

for c in ["Control", "Outputs", "Inputs"]:
    md_table.insert(1, c, ["?" for _ in range(len(md_table))])

for family, (inputs, outputs, control) in manual_data.items():
    md_table.loc[family, "Inputs"] = inputs
    md_table.loc[family, "Outputs"] = outputs
    md_table.loc[family, "Control"] = control

md_table.to_markdown(folder.joinpath("gym_table.md"), floatfmt=".3f", tablefmt="grid")
md_table.to_latex(folder.joinpath("gym_table.tex"), float_format="%.3f")

table_pdf = folder.joinpath("gym_table.pdf")
positions = [-md_table.index.get_loc(f) for f in sorted(df["Family"].unique())]

# Cancel this one to get something readable
#df.loc["CarRacing-v2"] = float("nan")

df.boxplot(column="Time", by="Family", vert=False, positions=positions)
plt.savefig(folder.joinpath("gym_table.png"))

df.boxplot(column="Time", by="Family", vert=False, positions=positions, figsize=(3, 2))
plt.title("")
plt.suptitle("")
#plt.xlabel("Time (s)")
plt.yticks([])
plt.ylabel("")
plt.xscale("log")
ax = plt.gca()
ax.tick_params(which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
ax.xaxis.set_label_position('top')
for y in [1, 2]:
    ax.axhline(y=-1*y-.5, linestyle="dashed", linewidth=.5, color="red")
for y, ay, dy, t, va in [(-1.5, -1.4, 1.5, "Faster &\nNo control", "bottom"),
                         (-2.6, -2.6, -2.5, "Slower &\nLow control", "top")]:
    ax.text(5.5, y, t, ha="right", va=va, size="x-small", color="red")
    plt.arrow(7, ay, 0, dy, color="red", head_width=1, head_length=.5, length_includes_head=True, clip_on=False)
ax.set_xlim(right=6)
for side in ['left', 'right', 'bottom']:
    ax.spines[side].set_visible(False)
plt.savefig(table_pdf, bbox_inches='tight', pad_inches=0.05)

with open(folder.joinpath("gym_pretty_table.tex"), "w") as f:
    lc = md_table.columns[-1]
    table_str = md_table.to_latex(float_format="%.3f")
    tr = table_str.split("\n")
    print("-"*80)
    print("\n".join(tr))

    tr[0] = tr[0][:-2] + "c@{}r@{}}"
    tr[2] = "&".join((r"\multirow{2}{*}{" + h + r"}" if 0 < i < 5 else h) for i, h in enumerate(tr[2].split("&")))
    tr[2] = tr[2].replace(r"Time \\", r"\multicolumn{2}{c}{Time (s)} \\ ")
    tr.insert(3, r"\cmidrule(lr){6-7}")
    tr[4] = "&".join(tr[5].split("&")[0:1]
                     + tr[4].split("&")[1:-1]
                     + [
                         r" Median & \multirow{6}{*}{"
                         + r"\includegraphics[height=\img]{"
                         + str(table_pdf) + r"}} \\"
                       ])
    del tr[5]
    tr[5] = r"\cmidrule(r){1-6}"
    tr[8] = " & ".join(r"\textbf{" + v + r"}" for v in tr[8].replace(r"\\", "").split(" & ")) + r"\\"

    print("-"*80)
    print("\n".join(tr))
    print("-"*80)

    def _print(*args): print(*args, file=f)
    _print(r"\documentclass{standalone}")
    _print(r"\usepackage{booktabs}")
    _print(r"\usepackage{multirow}")
    _print(r"\usepackage{tikz}")
    _print(r"\usetikzlibrary{calc, positioning}")
    _print(r"\newlength{\img}")
    _print(r"\setlength{\img}{7.5\baselineskip}")
    _print(r"\begin{document}")
    _print("\n".join(tr))
    _print(r"\end{document}")
