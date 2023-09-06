import seaborn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

X_ORDER = ["direct", "interpolation", "edhucat"]
HUE_ORDER = ["a2c", "ppo"]
SWARM_ARGS = dict(kind='swarm', palette='deep', dodge=True)
VIOLIN_ARGS = dict(kind='violin', inner=None, cut=0,
                   scale='width', palette='pastel')


def set_seaborn_style():  seaborn.set_style("darkgrid")


def move_legend(plot, ax=None, top=.95, hdl=None, lbl=None, **kwargs):
    l_args = dict(ncol=2, frameon=True, title=None,
                  handles=hdl, labels=lbl)
    if isinstance(plot, Figure):
        ax.legend(**l_args)
    else:
        seaborn.move_legend(obj=plot,
                            loc="lower center", bbox_to_anchor=(.5, 0),
                            **l_args)
    plt.tight_layout()
    plt.subplots_adjust(top=top)
