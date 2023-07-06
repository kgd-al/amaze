#!/usr/bin/env python3

import os.path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from scipy.interpolate import make_interp_spline, make_lsq_spline, make_smoothing_spline
from scipy.ndimage import gaussian_filter1d

if __name__ == "__main__":
    assert len(sys.argv) > 2, \
        f"Usage: {sys.argv[0]} csv-files[...]"

    files = sys.argv[1:]
    base = os.path.commonprefix(files)
    name = {Path(f).stem for f in files}
    assert len(name) == 1, f"Mismatching names. Expected one, got {name}"
    name = list(name)[0]

    data = []
    indices = set()
    for p in files:
        df = pd.read_csv(p, sep=' ', index_col=0)
        data.append(list(df[df.columns[0]]))
        indices |= set(df.index)

    indices = list(sorted(indices))

    def tolerant_mean(arrs):
        lens = [len(i) for i in arrs]
        arr = np.ma.empty((np.max(lens), len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[:len(l), idx] = list(l)
        return arr.mean(axis=-1), arr.std(axis=-1)

    fig, ax = plt.subplots()
    tr_avg, tr_std = tolerant_mean(data)
    df = pd.DataFrame(index=indices, data={"L": tr_avg-tr_std, "A": tr_avg, "U": tr_avg+tr_std})

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")

    ax.fill_between(indices, df.L, df.U, alpha=.5, label=r'$\pm$std')
    for d in [df.L, df.U]:
        ax.plot(indices, d, linewidth=.25, color="C0")

    for i, a in enumerate(data):
        ax.plot(indices[:len(a)], a,
                linestyle='dashed', linewidth=.2)

    i_spl = np.linspace(indices[0], indices[-1], 10*len(indices))

    ax.plot(indices, tr_avg, linewidth=.75, label='avg', linestyle='dashed')
    ax.plot(i_spl, make_smoothing_spline(indices, tr_avg)(i_spl), label='smooth')

    endpoints = [[indices[len(a)-1] for a in data],
                 [a[-1] for a in data]]
    for x, y in zip(*endpoints):
        ax.plot(x, y, marker='+')
    _ = ax.get_ylim()  # Just to get matplotlib to update its matrices

    fig.canvas.draw()
    # plt.gcf().canvas.draw()

    xc = {}
    texts = []
    gl_style = dict(linewidth=.5, color='gray', linestyle='dotted')
    bb_pad = .2
    for i, (x, y) in enumerate(zip(*endpoints)):
        x_, y_ = ax.transLimits.transform((x, y))
        ax.axvline(x, 0, y_, **gl_style)
        yc = .03
        t = ax.text(x_, yc, str(i),
                    horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(facecolor='white', #edgecolor='None',
                              boxstyle=f"square,pad={bb_pad}"),
                    transform=ax.transAxes)

        def get_bbox(text):
            # print(text.get_bbox_patch())
            bb: Bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
            # print(bb)
            p_ = bb_pad * 10
            bb.update_from_data_xy([[bb.x0 - p_, bb.y0 - p_],
                                    [bb.x1 + p_, bb.y1 + p_]],
                                   ignore=True)
            # print(bb)
            bb = bb.transformed(ax.transAxes.inverted())
            # print(bb)
            return bb

        tbb: Bbox = get_bbox(t)
        for t_ in texts:
            tbb_: Bbox = get_bbox(t_)
            i = tbb.overlaps(tbb_)
            # print(f"intersect({tbb}, {tbb_})")
            if i:
                t.set(y=tbb_.y0 + .03 + .2 * bb_pad)
                tbb = get_bbox(t)

        texts.append(t)

    ax.axline((0, ax.transLimits.transform((0, 1))[1]), slope=0, **gl_style)

    fig.legend(loc='outside upper center', ncols=3, borderaxespad=1)

    output = f"{base}{name}.png"
    fig.savefig(output, bbox_inches='tight')
    print("Generated", output)

    # train_data, test_data = [], []
    # train_indices, test_indices = set(), set()
    # for p in sys.argv[1:]:
    #     df_train = pd.read_csv(p.joinpath('stats.csv'), sep=' ', index_col=0)
    #     train_data.append(list(df_train['R']))
    #     train_indices |= set(df_train.index)
    #     df_test = pd.read_csv(p.joinpath('test_stats.csv'), sep=' ', index_col=0)
    #     test_data.append(list(df_test['Avg']))
    #     test_indices |= set(df_test.index)
    #
    # train_indices = list(sorted(train_indices))
    # test_indices = list(sorted(test_indices))
    # exit(0)
    #
    # def tolerant_mean(arrs):
    #     lens = [len(i) for i in arrs]
    #     arr = np.ma.empty((np.max(lens),len(arrs)))
    #     arr.mask = True
    #     for idx, l in enumerate(arrs):
    #         arr[:len(l),idx] = list(l)
    #     return arr.mean(axis = -1), arr.std(axis=-1)
    #
    # fig, ax = plt.subplots()
    # tr_avg, tr_std = tolerant_mean(train_data)
    # df = pd.DataFrame(index=train_indices, data={"L": tr_avg-tr_std, "A": tr_avg, "U": tr_avg+tr_std})
    # ax.fill_between(train_indices, df.L, df.U, alpha=.2)
    # for a in train_data:
    #     ax.plot(train_indices[:len(a)], a, alpha=.1)
    # ax.plot(train_indices, tr_avg)
    # for a in train_data:
    #     ax.plot(train_indices[len(a)-1], a[len(a)-1], marker='+')
    # fig.savefig(base.joinpath('$type-$trainer-stats.png'), bbox_inches='tight')
    #
    # fig, ax = plt.subplots()
    # te_avg, te_std = tolerant_mean(test_data)
    # df = pd.DataFrame(index=test_indices, data={"L": te_avg-te_std, "A": te_avg, "U": te_avg+te_std})
    # ax.fill_between(test_indices, df.L, df.U, alpha=.2)
    # for a in test_data:
    #     ax.plot(test_indices[:len(a)], a, alpha=.1)
    # ax.plot(test_indices, te_avg)
    # for a in test_data:
    #     ax.plot(test_indices[len(a)-1], a[len(a)-1], marker='+')
    # fig.savefig(base.joinpath('$type-$trainer-test-stats.png'), bbox_inches='tight')
