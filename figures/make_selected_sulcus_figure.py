"""Figure: how the depth of ONE selected sulcus is projected on the image.

The red circle drawn on the source image selects the sulcus; it is inpainted
away before segmentation so it is not counted as tissue.
Run from the repo root:  python figures/make_selected_sulcus_figure.py
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.getcwd())
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
from functions.measurements_image import _binarise_brain
from helpers.helpers import iter_hull_edge_sulci
from helpers.scalebar_reader import read_scalebar

SRC = "figures/2026-11-17 at 15.16.07.png"
OUT = "figures/sulcus_depth_projection_selected.png"
THRESH_MM = 0.15          # Sulcus Depth Threshold used for this figure

img = cv2.imread(SRC)
b, g, r = [img[:, :, i].astype(int) for i in range(3)]
red = ((r > 180) & (g < 110) & (b < 110)).astype(np.uint8)
ys, xs = np.nonzero(red)
cx, cy, R = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, (xs.max() - xs.min()) / 2
clean = cv2.inpaint(img, cv2.dilate(red, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)
mm = read_scalebar(clean).mm_per_px
bw = _binarise_brain(clean, cv2.COLOR_RGB2GRAY)
cnt = max(cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
pts = cnt[:, 0, :].astype(float)
hull = cv2.convexHull(cnt)[:, 0, :]

edges = list(iter_hull_edge_sulci(cnt, THRESH_MM / mm))
best = None                                   # sulcus nearest to the user's circle
for s, e, sul in edges:
    for p, d in sul:
        dd = np.hypot(p[0] - cx, p[1] - cy)
        if best is None or dd < best[0]:
            best = (dd, s, e, sul, p, d)
_, S, E, SUL, SEL, SEL_D = best
a, bb = np.array(S, float), np.array(E, float); ab = bb - a
i_s = int(np.argmin(np.hypot(*(pts - a).T))); i_e = int(np.argmin(np.hypot(*(pts - bb).T)))
idx = np.arange(i_s, i_e + 1) if i_e >= i_s else np.r_[np.arange(i_s, len(pts)), np.arange(0, i_e + 1)]
seg = pts[idx]
foot = lambda p: a + np.dot(np.asarray(p, float) - a, ab) / np.dot(ab, ab) * ab

C_CNT, C_HULL, C_S, C_D, C_O = "#ff3b30", "#4aa3ff", "#2ee66b", "#ff9f0a", "#9aa0a6"
H, W = clean.shape[:2]; k = 0.3
small = cv2.cvtColor(cv2.resize(clean, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
ext = (0, W, H, 0)

def show(ax, x0, x1, y0, y1):
    ax.imshow(small, extent=ext, interpolation="bilinear"); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")

def right_angle(ax, f, p, q):
    u = ab / np.linalg.norm(ab); v = (np.asarray(p, float) - f); v /= np.linalg.norm(v)
    ax.plot([f[0] + u[0] * q, f[0] + (u + v)[0] * q, f[0] + v[0] * q],
            [f[1] + u[1] * q, f[1] + (u + v)[1] * q, f[1] + v[1] * q], color=C_D, lw=1.6, zorder=6)

fig = plt.figure(figsize=(17, 7.4), dpi=150)
axA = fig.add_axes([0.005, 0.07, 0.31, 0.80]); axB = fig.add_axes([0.335, 0.07, 0.33, 0.80]); axC = fig.add_axes([0.70, 0.30, 0.285, 0.57])

# ---- A overview
show(axA, 650, 5100, 750, 3900)
axA.plot(*np.r_[pts, pts[:1]].T, color=C_CNT, lw=1.3)
axA.plot(*np.r_[hull, hull[:1]].T, color=C_HULL, lw=1.8)
axA.plot([a[0], bb[0]], [a[1], bb[1]], color=C_D, lw=3.5, zorder=4)
axA.add_patch(Circle(SEL, 210, fill=False, ec=C_D, lw=2.6, zorder=6))
axA.scatter(*SEL, s=45, color=C_S, edgecolor="white", lw=.8, zorder=7)
bx0, bx1, by0, by1 = 2150, 3750, 800, 2200
axA.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec="white", lw=1.4, ls="--", zorder=5))
axA.set_title("A  Coronal slice and its convex hull", loc="left", fontsize=13, fontweight="bold")

# ---- B hull edge zoom
show(axB, bx0, bx1, by0, by1)
axB.plot(*seg.T, color=C_CNT, lw=2.4, zorder=3)
axB.plot([a[0], bb[0]], [a[1], bb[1]], color=C_HULL, lw=3, zorder=4)
for P, lab, off in ((a, "s", (14, -16)), (bb, "e", (-14, -16))):
    axB.scatter(*P, s=90, color=C_HULL, edgecolor="white", lw=1.3, zorder=8)
    axB.annotate(lab, P, xytext=off, textcoords="offset points", fontsize=16, fontweight="bold", color=C_HULL, ha="center", va="center", zorder=9)
for p, d in SUL:
    f = foot(p)
    if np.hypot(p[0] - SEL[0], p[1] - SEL[1]) < 1:
        continue
    axB.plot([p[0], f[0]], [p[1], f[1]], color=C_O, lw=1.8, ls=(0, (4, 3)), zorder=4)
    axB.scatter(*p, s=40, color=C_O, edgecolor="white", lw=.8, zorder=6)
    axB.annotate(f"{d * mm:.2f} mm", (p[0], p[1]), xytext=(8, 10), textcoords="offset points", fontsize=10, color="#dddddd", zorder=9)
f = foot(SEL)
axB.plot([SEL[0], f[0]], [SEL[1], f[1]], color=C_D, lw=3, zorder=7)
axB.scatter(*f, s=40, color=C_D, edgecolor="white", lw=.8, zorder=8)
axB.scatter(*SEL, s=120, color=C_S, edgecolor="white", lw=1.5, zorder=9)
right_angle(axB, f, SEL, 70)
mid = (np.array(SEL) + f) / 2
axB.annotate(f"depth = {SEL_D * mm:.2f} mm\n({SEL_D:.0f} px)", mid, xytext=(18, 0), textcoords="offset points",
             fontsize=12, fontweight="bold", va="center", ha="left", zorder=10,
             bbox=dict(boxstyle="round,pad=.35", fc="white", ec=C_D, lw=1.8))
axB.set_title("B  The hull edge s–e and the selected sulcus", loc="left", fontsize=13, fontweight="bold")
tx0, tx1, ty0, ty1 = SEL[0] - 230, SEL[0] + 230, SEL[1] - 190, SEL[1] + 190
axB.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False, ec="white", lw=1.4, ls="--", zorder=6))

# ---- C close-up of the tip
show(axC, tx0, tx1, ty0, ty1)
axC.plot(*seg.T, color=C_CNT, lw=3, zorder=3)
axC.plot([SEL[0], f[0]], [SEL[1], f[1]], color=C_D, lw=3.5, zorder=5)
axC.scatter(*SEL, s=170, color=C_S, edgecolor="white", lw=1.8, zorder=7)
axC.annotate("sulcus point\n(deepest point of\nthe slit)", SEL, xytext=(-95, -62), textcoords="offset points", fontsize=11.5, color="white",
             ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4), zorder=9)
axC.annotate("perpendicular to\nthe hull chord  ↑", (SEL[0], SEL[1] - 110), xytext=(-12, 20), textcoords="offset points", fontsize=11, color=C_D,
             ha="right", va="center", fontweight="bold", zorder=9)
axC.set_title("C  Close-up of the slit tip", loc="left", fontsize=13, fontweight="bold")

for yy in (0.0, 1.0):                         # dashed leaders between panels
    fig.add_artist(ConnectionPatch(xyA=(bx1, by0 if yy else by1), coordsA=axA.transData, xyB=(0, yy), coordsB=axB.transAxes, color="#888", lw=.9, ls="--"))
    fig.add_artist(ConnectionPatch(xyA=(tx1, ty0 if yy else ty1), coordsA=axB.transData, xyB=(0, yy), coordsB=axC.transAxes, color="#888", lw=.9, ls="--"))

h = [plt.Line2D([], [], color=C_CNT, lw=2.5), plt.Line2D([], [], color=C_HULL, lw=3), plt.Line2D([], [], color=C_D, lw=3),
     plt.Line2D([], [], marker="o", color="w", mfc=C_S, mec="white", ms=11, lw=0)]
fig.legend(h, ["Contour between s and e", "Hull chord  s–e", "Depth = perpendicular distance from sulcus point to chord", "Selected sulcus point"],
           loc="lower left", bbox_to_anchor=(0.70, 0.065), fontsize=10.5, frameon=False, labelspacing=.9)
fig.text(0.01, 0.955, "How the depth of the selected sulcus is measured on the image", fontsize=16, fontweight="bold", va="center")
fig.text(0.01, 0.025, f"{os.path.basename(SRC)}   |   {mm:.4f} mm/px (2 mm scalebar)   |   red selection circle removed before analysis   |   "
         f"Sulcus Depth Threshold {THRESH_MM:g} mm (at the 1 mm default this slit is merged into the large fissure and not counted separately)",
         fontsize=8.8, color="#444")
fig.savefig(OUT, facecolor="white"); print(OUT, "| selected sulcus", SEL, "depth_mm=%.3f" % (SEL_D * mm), "| sulci on edge:", [(p, round(d * mm, 2)) for p, d in SUL])
