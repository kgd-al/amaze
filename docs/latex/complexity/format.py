import math
from pathlib import Path
from random import Random

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from amaze.simu.types import MazeClass

df_file = Path(__file__).parent.joinpath('data.csv')
df = pd.read_csv(df_file, index_col=0)

print(df)
hue_order = [c.name.capitalize() for c in list(reversed(MazeClass))[1:]]

if False:
    kwargs = dict(x="Surprisingness", y="Deceptiveness", hue="Class", hue_order=hue_order)

    sns.set_style("darkgrid")
    g = sns.jointplot(data=df, **kwargs,
                      kind="kde", cut=0, levels=10, thresh=.01, bw_adjust=1,
                      fill=True, alpha=.5,
                      marginal_kws=dict(
                          multiple="stack", alpha=.25, linewidth=.1,
                          bw_adjust=1,
                          warn_singular=False
                      ),
                      warn_singular=False)
    g.ax_joint.legend_.set_title("Maze class")
    g.ax_joint.autoscale(enable=True, tight=True)

    sns.scatterplot(data=df.sample(n=100, random_state=0), **kwargs, size=.1,
                    ax=g.ax_joint, legend=False)

    plt.savefig('foo.pdf')

if True:
    def eqd(a, b): return math.sqrt(a*a+b*b)

    for m_class, func in [("Simple", "idxmin"), ("Traps", idxmin)]
    df[df.Class == "Simplp"].apply(axis=1, func=lambda xs: eqd(*xs[1:])).idxmin()
