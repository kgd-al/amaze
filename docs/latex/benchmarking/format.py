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
plt.gca().tick_params(which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
plt.gca().xaxis.set_label_position('top')
plt.savefig(table_pdf, bbox_inches='tight')

with open(folder.joinpath("gym_pretty_table-auto.tex"), "w") as f:
    lc = md_table.columns[-1]
    table_str = md_table.to_latex(float_format="%.3f")
    table_str = table_str.replace(r"lr}", r"lrr}")
    table_str = table_str.replace("Time", f"\\multicolumn{{2}}{{c}}{{Time (s)}}")
    table_str = table_str.replace("median", r"median & \multirow{6}{*}{"
                                            + r"\includegraphics[height=6\baselineskip]{"
                                            + str(table_pdf) + r"}}")

    def _print(*args): print(*args, file=f)
    _print(r"\documentclass{standalone}")
    _print(r"\usepackage{booktabs}")
    _print(r"\usepackage{multirow}")
    _print(r"\usepackage{tikz}")
    _print(r"\usetikzlibrary{calc, positioning}")
    _print(r"\begin{document}")
    #_print(r"\begin{tikzpicture}")
    #_print(r" \node (T){")
    _print(table_str)
    #_print(r" };")
    #_print(r" \path let \p{A} = ($(T.north)-(T.south)$) in")
    #_print(r"  node (I) [above right=0 of T.south east, xshift=-1cm, inner sep=0pt, draw]")
    #_print(r"   {\includegraphics[height=6\baselineskip]{", table_pdf, r"}};")
    #_print(r"\end{tikzpicture}")
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
