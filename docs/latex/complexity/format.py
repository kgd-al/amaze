import math
import pprint
import sys
from pathlib import Path
from random import Random

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from scipy import spatial

import pandas as pd

import matplotlib.image as image
import seaborn as sns

from amaze import amaze_main
from amaze.simu.types import MazeClass

plt.rcParams["image.composite_image"] = False
dpi = 300

root = Path(__file__).parent
df_file = root.joinpath('data.csv')
df = pd.read_csv(df_file, index_col=0)

print(df)
hue_order = [c.name.capitalize() for c in list(reversed(MazeClass))[1:]]

if len(sys.argv) > 1:
    print("Fast-mode: sampling 10000 points from the data")
    df = df.sample(10000, random_state=1)

# Sample points for readable scatterplot
# For Trivial and Simple: get the min-max along S (D always equals 0)
# For Lures, Traps and Complex: get the convex hull
# Then complement as needed
sample_size = 1000
sample_df = df[df.Class == "NOTHING"]
for m_class in hue_order:
    c_df = df[df.Class == m_class]
    if len(c_df["Deceptiveness"].unique()) == 1:
        hull_idx = c_df.Surprisingness.agg(['idxmin', 'idxmax'])
    else:
        hull_idx = spatial.ConvexHull(c_df[["Surprisingness", "Deceptiveness"]]).vertices
        hull_idx = c_df.index[hull_idx]
    hull_df = c_df.loc[hull_idx]
    c_df = c_df.drop(hull_idx)
    sample_df = pd.concat([
        sample_df, hull_df,
        c_df.sample(sample_size // 5 - len(hull_df), random_state=0)
    ])

print("Sample:", sample_df)

# Plot kde of data + extracted sample
kwargs = dict(x="Surprisingness", y="Deceptiveness", hue="Class", hue_order=hue_order)
sns.set_style("darkgrid")
g = sns.jointplot(data=df, **kwargs,
                  kind="kde", cut=0, levels=10, thresh=.01, bw_adjust=.7,
                  fill=True, alpha=1,
                  marginal_kws=dict(
                      multiple="stack", alpha=.5, linewidth=.1,
                      bw_adjust=.8,
                      warn_singular=False
                  ),
                  warn_singular=False)
g.ax_joint.legend_.set_title("Maze class")
g.ax_joint.autoscale(enable=True, tight=True)

sns.scatterplot(data=sample_df, **kwargs, s=5,
                ax=g.ax_joint, legend=False)


# Find extremum in each class to showcase examples
def eqd(x, y): return math.sqrt(x**2+y**2)
def process(_fn): return lambda a: _fn(a["Surprisingness"], a["Deceptiveness"])


examples = root.joinpath('examples')
examples.mkdir(parents=True, exist_ok=True)

fs = 6
g.figure.set_size_inches(2*fs, fs)
plt.subplots_adjust(left=.25, right=.75)

for i, m_class, fn, bxy, zoom in [
            (0, "Complex", lambda x, y: eqd(x, y), (1, .75), (.75, .75, 225)),
            (1, "Traps", lambda _, y: y, (0, .75), ()),
            (2, "Lures", lambda x, _: x, (1, .25), ()),
            (4, "Trivial", lambda x, y: -eqd(x, y), (0, .25), ()),
        ]:

    c_df = sample_df[sample_df.Class == m_class]
    m = c_df.apply(process(fn), axis=1).idxmax()
    output = examples.joinpath(f"{m_class.lower()}__{m}.png")
    if not output.exists():
        amaze_main(f"--maze {m} --render {output} --dark --colorblind --width 500")

    img = Image.open(output)
    target_size = 2 * fs * dpi // 5
    img_box = OffsetImage(img.resize((target_size, target_size)), zoom=72./dpi)
    xy = c_df.loc[m][["Surprisingness", "Deceptiveness"]]
    b_anchor = (bxy[0], .5)
    ann_box = AnnotationBbox(img_box, xy=xy, xycoords='data',
                             xybox=bxy, boxcoords="figure fraction",
                             arrowprops=dict(arrowstyle="-",
                                             linestyle=(0, (5, 10)),
                                             color=f"C{i}"),
                             frameon=False,
                             box_alignment=b_anchor,
                             annotation_clip=False)
    g.ax_joint.add_artist(ann_box)

    if len(zoom) > 0:
        zx, zy, za = zoom
        zxy = (bxy[0] + zx * .1 - .1, bxy[1] + zy * .1)
        cs = .1

        clip_width = 1.4 * cs
        circ = patches.Ellipse((zxy[0] - .0135,
                                zxy[1] + .01),
                               clip_width, clip_width * 2,
                               transform=g.figure.transFigure)

        img_zoom_box = OffsetImage(img.crop(((zx - cs) * img.width,
                                             (zy - cs) * img.height,
                                             (zx + cs) * img.width,
                                             (zy + cs) * img.height)),
                                   zoom=5*72./dpi)#, clip_path=circ)
        # img_zoom_box.set_clip_path(circ)
        # img_zoom_box.set_clip_on(True)

        ann_ann_box = AnnotationBbox(img_zoom_box, xy=zxy,
                                     xycoords='figure fraction',
                                     frameon=True,
                                     bboxprops=dict(
                                         boxstyle="Circle", facecolor="none",
                                         edgecolor="red", linewidth=5,
                                     ),
                                     pad=0,
                                     annotation_clip=False)
        
        g.ax_joint.add_artist(ann_ann_box)

# All done. Save
# g.figure.tight_layout()
g.figure.savefig('foo.pdf', bbox_inches='tight', dpi=dpi)
g.figure.savefig('foo.png', bbox_inches='tight', dpi=dpi)

