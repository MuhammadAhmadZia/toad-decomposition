"""
Methodology diagram for:
  "A Four-Factor Decomposition of Memory Compression in Boosted Tree Ensembles"

Writes methodology.png (300 dpi) and methodology.pdf (vector; prefer the PDF for LaTeX).
Pure matplotlib, no external dependencies. Restyle from the CONFIG block below.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
FIG_W, FIG_H = 7.16, 5.1           # IEEE full width, inches
FS_HEAD, FS_BODY, FS_SMALL = 9.2, 8.0, 6.9
FONT = "DejaVu Sans"

# Colour encodes WHEN a factor acts, which is the paper's core distinction:
# two factors change the model, one edits it afterwards, one only changes
# how an otherwise unchanged model is counted.
C_TRAIN   = "#2C5F8D"
C_POST    = "#B5533C"
C_ACCOUNT = "#3F7A5E"
C_NEUTRAL = "#454545"
C_LIGHT   = "#F4F4F4"
C_EDGE    = "#9A9A9A"

plt.rcParams.update({"font.family": FONT})
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

LINE_H, PAD_T, PAD_H, PAD_B = 4.4, 4.8, 5.2, 3.2


def box_h(n_lines):
    return PAD_T + PAD_H + LINE_H * max(n_lines - 1, 0) + PAD_B


def box(x, y, w, title, lines, edge, fill="white", lw=1.15, fs_t=FS_BODY,
        min_lines=0):
    """y is the BOTTOM edge. Height follows the content, so nothing overflows."""
    h = box_h(max(len(lines), min_lines))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.5",
                                linewidth=lw, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - PAD_T, title, ha="center", va="center",
            fontsize=fs_t, color=edge, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - PAD_T - PAD_H - i * LINE_H, ln,
                ha="center", va="center", fontsize=FS_SMALL, color="#333333")
    return h


def arrow(x1, y1, x2, y2, colour=C_NEUTRAL, lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=lw,
                                 color=colour, shrinkA=0, shrinkB=0))


# ---------------- content ----------------
F = [("F1  Learner",        ["LightGBM (leaf-wise)", "CatBoost (oblivious)"],            C_TRAIN),
     ("F2  Model size",     ["num_leaves, max_depth", "n_estimators, max_bin"],          C_TRAIN),
     ("F3  Storage layout", ["fp32 / fp16 arrays", "implicit tree array",
                             "succinct (LOUDS)"],                                        C_ACCOUNT),
     ("F4  Value coding",   ["threshold clustering", "leaf clustering",
                             "leaf quantization"],                                       C_POST)]

N_LINES = max(len(f[1]) for f in F)
H_IN, H_F = box_h(1), box_h(N_LINES)
H_M, H_A = box_h(2), box_h(2)
G1, G_LBL, G_BUS, G3 = 6.0, 7.0, 5.0, 6.0

y = 96.0                                     # top of the drawing
Y_IN = y - H_IN
Y_F  = Y_IN - G1 - H_F
Y_BUS = Y_F - G_LBL
Y_M  = Y_BUS - G_BUS - H_M
Y_A  = Y_M - G3 - H_A

XS, FW = [4.0, 25.5, 47.0, 68.5], 18.0

# ---------------- legend ----------------
handles = [Line2D([], [], color=C_TRAIN, lw=2.6, label="acts during training"),
           Line2D([], [], color=C_POST, lw=2.6, label="edits a fitted model"),
           Line2D([], [], color=C_ACCOUNT, lw=2.6, label="representation only")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.055),
          fontsize=FS_SMALL, frameon=False, handlelength=1.6,
          ncol=3, columnspacing=2.2)

# ---------------- 1. input ----------------
box(4, Y_IN, 82, "Tabular benchmarks",
    ["5 datasets   ·   12 seeds   ·   22,800 fitted models"],
    C_EDGE, fill=C_LIGHT, fs_t=FS_HEAD)

# ---------------- 2. the four factors ----------------
for x, (title, lines, col) in zip(XS, F):
    arrow(x + FW / 2, Y_IN, x + FW / 2, Y_F + H_F)
    box(x, Y_F, FW, title, lines, col, min_lines=N_LINES)

ax.text(XS[0] + FW + 2.25, Y_F - 3.2, "change the fitted model", ha="center",
        va="center", fontsize=FS_SMALL, color=C_TRAIN, style="italic")
ax.text(XS[2] + FW / 2, Y_F - 3.2, "model unchanged", ha="center",
        va="center", fontsize=FS_SMALL, color=C_ACCOUNT, style="italic")
ax.text(XS[3] + FW / 2, Y_F - 3.2, "post-hoc edit", ha="center",
        va="center", fontsize=FS_SMALL, color=C_POST, style="italic")

ax.plot([XS[0] + 1.5, XS[0] + 1.5, XS[1] + FW - 1.5, XS[1] + FW - 1.5],
        [Y_F - 0.9, Y_F - 1.9, Y_F - 1.9, Y_F - 0.9], color=C_TRAIN, lw=0.9)

# ---------------- 3. merge bus ----------------
for x, (_, _, col) in zip(XS, F):
    ax.plot([x + FW / 2, x + FW / 2], [Y_F - 5.4, Y_BUS], color=col, lw=1.1)
ax.plot([XS[0] + FW / 2, XS[3] + FW / 2], [Y_BUS, Y_BUS], color=C_NEUTRAL, lw=1.1)
ax.text(XS[3] + FW / 2 + 3.0, Y_BUS, "one scored\nconfiguration", ha="left",
        va="center", fontsize=FS_SMALL, color=C_NEUTRAL, style="italic")

# ---------------- 4. two measurements ----------------
arrow(26.0, Y_BUS, 28.0, Y_M + H_M)
arrow(65.0, Y_BUS, 72.0, Y_M + H_M)
box(4.0, Y_M, 44.0, "Memory accounting",
    ["bit-exact reimplementation of the", "published storage layout"],
    C_NEUTRAL, fs_t=FS_BODY)
box(43.0, Y_M, 44.0, "Predictive quality",
    ["held-out accuracy or $R^2$, verified", "against the library's own output"],
    C_NEUTRAL, fs_t=FS_BODY)

# ---------------- 5. analysis ----------------
arrow(26.0, Y_M, 36.0, Y_A + H_A)
arrow(65.0, Y_M, 55.0, Y_A + H_A)
box(16.0, Y_A, 59.0, "Iso-quality compression ratio",
    ["memory needed to reach a fixed quality level,",
     "each learner priced against its own fp32 baseline"],
    C_NEUTRAL, fill=C_LIGHT, fs_t=FS_BODY)

plt.tight_layout(pad=0.3)
fig.savefig("methodology.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("methodology.pdf", bbox_inches="tight", facecolor="white")
print(f"wrote methodology.png / .pdf   (lowest element at y={Y_A:.1f})")
