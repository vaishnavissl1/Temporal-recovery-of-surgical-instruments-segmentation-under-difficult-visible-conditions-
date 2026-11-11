#!/usr/bin/env python3
"""
Architecture diagram of the system AS BUILT (deck slide 5): plain box-and-arrow style, 300 dpi.

    python -m tools.arch_figure --out "C:/Users/Chandaluri Vaishnavi/Downloads/cvrsfesp/cvavish/completearch.png"

Shows: the stereo pair; base segmentation on the left view; the reliability gate (nine observable indicators); the decision; the branch where reliable frames pass through unmodified; the memory buffer
with its own admission threshold; candidate actions generated on flagged frames only (temporal A2, refinement A3); the selector with its abstain option; identity verification on both paths; the final
mask. The right-view path is drawn as an audit path, dashed and marked evaluated-but-not-adopted, because the cross-view indicators did not pass their pre-set test and are not in the shipped gate.
It deliberately shows NO difficult-event construction stage and NO downstream tracking stage: neither exists. The retired actions (stereo label transfer and its fusion) are not drawn.
The canvas is 9.583 x 3.565 in, the size of the picture placeholder on the slide, so 10 pt text here is 10 pt on the slide.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402

W, H = 9.583, 3.565
FS = 10
X = [0.95, 2.85, 4.75, 6.65, 8.55]       # five columns
Y = {"A": 3.0, "B": 1.75, "C": 0.5}
BW, BH = 1.62, 0.78


def box(ax, x, y, text, dashed=False, w=BW, h=BH, bold=False):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=True, fc="#ffffff" if not dashed else "#f2f2f2", ec="#1a1a2e", lw=1.4, ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, color="#111111", fontweight="bold" if bold else "normal", linespacing=1.18)


def diamond(ax, x, y, text, w=1.5, h=0.9):
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)], closed=True, fc="#ffffff", ec="#1a1a2e", lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS)


def arrow(ax, p, q, dashed=False, conn="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.5, color="#1a1a2e", ls="--" if dashed else "-", connectionstyle=conn, shrinkA=0, shrinkB=0))


def line(ax, pts, dashed=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#1a1a2e", lw=1.5, ls="--" if dashed else "-", solid_capstyle="butt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="completearch.png")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # row A: the main path
    box(ax, X[0], Y["A"], "Stereo pair\nleft and right frame")
    box(ax, X[1], Y["A"], "Base segmentation\n(left view)")
    box(ax, X[2], Y["A"], "Reliability gate\n9 observable\nindicators", h=0.95)
    diamond(ax, X[3], Y["A"], "reliable?\n$r_t \\geq \\tau$")
    box(ax, X[4], Y["A"], "Pass through\nunmodified")
    for p, q in ((X[0] + BW / 2, X[1] - BW / 2), (X[1] + BW / 2, X[2] - BW / 2), (X[2] + BW / 2, X[3] - 0.75)):
        arrow(ax, (p, Y["A"]), (q, Y["A"]))
    arrow(ax, (X[3] + 0.75, Y["A"]), (X[4] - BW / 2, Y["A"]))
    ax.text(X[3] + 0.78, Y["A"] + 0.1, "yes", fontsize=9, ha="left", va="bottom")
    # row B: flagged path
    box(ax, X[2], Y["B"], "Memory buffer\nlast K frames,\nown threshold $\\tau_{mem}$", h=0.95)
    box(ax, X[3], Y["B"], "Candidate actions\nA2 temporal\nA3 refinement", h=0.95)
    box(ax, X[4], Y["B"], "Selector\npredicts gains,\nmay abstain", h=0.95)
    arrow(ax, (X[3], Y["A"] - 0.45), (X[3], Y["B"] + 0.43))
    ax.text(X[3] + 0.06, (Y["A"] + Y["B"]) / 2 + 0.05, "no: flagged", fontsize=9, ha="left", va="center")
    arrow(ax, (X[2] + BW / 2, Y["B"]), (X[3] - BW / 2, Y["B"]))
    arrow(ax, (X[3] + BW / 2, Y["B"]), (X[4] - BW / 2, Y["B"]))
    # memory is written from the pass-through frames and the gate output (admission by its own threshold)
    arrow(ax, (X[2] + 0.35, Y["A"] - 0.43), (X[2] + 0.35, Y["B"] + 0.47), dashed=True)
    ax.text(X[2] + 0.42, (Y["A"] + Y["B"]) / 2 + 0.02, "store if $r_t \\geq \\tau_{mem}$", fontsize=9, ha="left", va="center")
    # row C: identity and output
    box(ax, X[4], Y["C"], "Identity\nverification")
    box(ax, X[3], Y["C"], "Final mask")
    arrow(ax, (X[4], Y["B"] - 0.43), (X[4], Y["C"] + 0.39))
    line(ax, [(X[4] + BW / 2, Y["A"]), (9.45, Y["A"]), (9.45, Y["C"])])
    arrow(ax, (9.45, Y["C"]), (X[4] + BW / 2, Y["C"]))
    arrow(ax, (X[4] - BW / 2, Y["C"]), (X[3] + BW / 2, Y["C"]))
    # right-view audit (dashed: evaluated, not adopted)
    box(ax, X[1], Y["B"], "Right-view audit\n(evaluated,\nnot adopted)", dashed=True, h=0.95)
    line(ax, [(X[0], Y["A"] - 0.39), (X[0], Y["B"]), (X[1] - BW / 2, Y["B"])], dashed=True)
    arrow(ax, (X[1] - BW / 2, Y["B"]), (X[1] - BW / 2 - 0.001, Y["B"]), dashed=True)
    line(ax, [(X[1], Y["B"] + 0.47), (X[1], 2.3), (X[2] - 0.4, 2.3)], dashed=True)
    arrow(ax, (X[2] - 0.4, 2.3), (X[2] - 0.4, Y["A"] - 0.43), dashed=True)
    fig.savefig(a.out, dpi=300, facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
