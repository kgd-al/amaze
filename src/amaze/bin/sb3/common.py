import seaborn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

X_ORDER = ["direct", "interpolation", "edhucat"]
HUE_ORDER = ["a2c", "ppo"]
SWARM_ARGS = dict(kind='swarm', palette='deep', dodge=True,
                  s=3.5)
VIOLIN_ARGS = dict(kind='violin', inner=None, cut=0,
                   scale='width', palette='pastel')

FIG_SIZE_INCHES = (7, 3.5)


def set_seaborn_style():
    seaborn.set_style("darkgrid")
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams["figure.figsize"] = FIG_SIZE_INCHES


def move_legend(plot, ax=None, top=.95, hdl=None, lbl=None, **kwargs):
    l_args = dict(ncol=2, frameon=True, title=None,
                  handles=hdl, labels=lbl, **kwargs)
    if isinstance(plot, Figure):
        ax.legend(**l_args)
    else:
        seaborn.move_legend(obj=plot,
                            loc="upper center", bbox_to_anchor=(.5, 1.05),
                            **l_args)
    plt.tight_layout()
    plt.subplots_adjust(top=top)
