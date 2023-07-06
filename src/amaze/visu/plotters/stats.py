from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_stats(folder, train_stats, test_stats):
    fig, ax = plt.subplots()
    fig: Figure = fig
    ax: Axes = ax
    ax.plot(train_stats["R"])
    fig.savefig(folder.joinpath("stats.png"), bbox_inches='tight')
    train_stats.to_csv(folder.joinpath("stats.csv"), sep=' ')

    fig, ax = plt.subplots()
    ax.fill_between(test_stats.index,
                    test_stats["Avg"]-test_stats["Std"],
                    test_stats["Avg"]+test_stats["Std"],
                    alpha=0.2)
    ax.plot(test_stats["Avg"])
    fig.savefig(folder.joinpath("test_stats.png"),
                bbox_inches='tight')
    test_stats.to_csv(folder.joinpath("test_stats.csv"), sep=' ')
