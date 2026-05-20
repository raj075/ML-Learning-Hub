import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# ── colour palette ────────────────────────────────────────────────────────────
C_TITLE  = "#1a1a2e"
C_DATA   = "#16213e"
C_BOOT   = "#0f3460"
C_DT     = "#533483"
C_VOTE   = "#e94560"
C_ARROW  = "#555555"
C_PRED0  = "#e94560"
C_PRED1  = "#2ecc71"
C_NOTE   = "#777777"

def rounded_box(ax, xy, w, h, color, alpha=0.12, lw=2, ls="-"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         linewidth=lw, linestyle=ls,
                         edgecolor=color,
                         facecolor=color, alpha=alpha)
    ax.add_patch(box)
    return box

def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14))

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax.text(7, 8.55, "Bagging  ·  Random Forest",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color=C_TITLE, fontfamily="monospace")
ax.axhline(8.3, xmin=0.04, xmax=0.96, color=C_TITLE, lw=1.2)

# ── ORIGINAL DATASET ──────────────────────────────────────────────────────────
rounded_box(ax, (0.4, 4.0), 2.6, 3.6, C_DATA, alpha=0.10, lw=2.5)
ax.text(1.7, 7.2, "Dataset", ha="center", va="center",
        fontsize=11, fontweight="bold", color=C_DATA)
ax.text(1.7, 6.55, "M × N", ha="center", va="center",
        fontsize=16, fontweight="bold", color=C_DATA, fontstyle="italic")
ax.text(1.7, 5.95, "300 rows", ha="center", va="center",
        fontsize=10, color=C_DATA)
ax.text(1.7, 5.55, "N  features", ha="center", va="center",
        fontsize=10, color=C_DATA)
ax.text(1.7, 5.05, "labels: {0, 1}", ha="center", va="center",
        fontsize=9, color=C_DATA, style="italic")

# sampling notes below dataset box
ax.text(1.7, 3.55, "Row sampling:", ha="center", fontsize=8.5, color=C_NOTE)
ax.text(1.7, 3.15, "with replacement", ha="center", fontsize=8.5,
        color=C_NOTE, fontstyle="italic")
ax.text(1.7, 2.65, "Feature sampling:", ha="center", fontsize=8.5, color=C_NOTE)
ax.text(1.7, 2.25, r"$m = \sqrt{N}$  (clf)  or  $N/3$  (reg)",
        ha="center", fontsize=8.5, color=C_NOTE, fontstyle="italic")

# ── BOOTSTRAP SAMPLES + DECISION TREES ───────────────────────────────────────
configs = [
    # (y_centre, bootstrap label, m, n, prediction, pred_color)
    (6.8, "B₁", "m₁", "n₁", "→  1", C_PRED1),
    (5.1, "B₂", "m₂", "n₂", "→  0", C_PRED0),
    (3.4, "B₃", "m₃", "n₃", "→  1", C_PRED1),
]

for (yc, blabel, ml, nl, pred, pc) in configs:
    bx, by = 4.6, yc - 0.65

    # Bootstrap box
    rounded_box(ax, (bx, by), 1.3, 1.3, C_BOOT, alpha=0.13, lw=1.8)
    ax.text(bx + 0.65, yc + 0.1, blabel,
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=C_BOOT)
    ax.text(bx + 0.65, yc - 0.35,
            f"{ml} rows\n{nl} feats",
            ha="center", va="center", fontsize=8, color=C_BOOT)

    # Decision Tree box
    dtx = bx + 2.1
    rounded_box(ax, (dtx, by), 1.4, 1.3, C_DT, alpha=0.13, lw=1.8)
    ax.text(dtx + 0.7, yc + 0.12, "Decision",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=C_DT)
    ax.text(dtx + 0.7, yc - 0.28, "Tree",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=C_DT)

    # Prediction label
    ax.text(dtx + 1.8, yc - 0.08, pred,
            ha="left", va="center", fontsize=13,
            fontweight="bold", color=pc)

    # arrow: dataset → bootstrap
    arrow(ax, 3.0, yc - 0.0, bx + 0.05, yc - 0.0, color=C_BOOT, lw=1.4)
    # arrow: bootstrap → DT
    arrow(ax, bx + 1.3, yc - 0.0, dtx + 0.05, yc - 0.0, color=C_DT, lw=1.4)

# ── MAJORITY VOTE / AGGREGATION ───────────────────────────────────────────────
vx, vy = 10.3, 4.55
rounded_box(ax, (vx, vy), 2.4, 2.0, C_VOTE, alpha=0.12, lw=2.2)
ax.text(vx + 1.2, vy + 1.65, "Aggregate",
        ha="center", va="center", fontsize=11,
        fontweight="bold", color=C_VOTE)
ax.text(vx + 1.2, vy + 1.1,
        "Classification:\nmajority vote\n\nRegression:\nmean",
        ha="center", va="center", fontsize=9, color=C_VOTE)

# arrows: predictions → aggregate
for (yc, *_) in configs:
    arrow(ax, 9.25, yc - 0.0, vx + 0.05, 5.55, color=C_VOTE, lw=1.2)

# ── FINAL OUTPUT ─────────────────────────────────────────────────────────────
arrow(ax, vx + 2.4, 5.55, 13.0, 5.55, color=C_VOTE, lw=2.0)
ax.text(13.1, 5.55, "1",
        ha="left", va="center", fontsize=22,
        fontweight="bold", color=C_PRED1)
ax.text(13.1, 5.05, "(2 vs 1)",
        ha="left", va="center", fontsize=8, color=C_NOTE)

# ── LEGEND / KEY NOTES ───────────────────────────────────────────────────────
note_lines = [
    r"$\bf{Bootstrap}$: sample rows with replacement",
    r"$\bf{Feature\ sampling}$: $\sqrt{N}$ features per split (classification)",
    r"$\bf{Diversity}$: each tree sees different data & features",
    r"$\bf{Parallelism}$: all trees train independently",
]
for i, line in enumerate(note_lines):
    ax.text(0.5, 1.65 - i * 0.42, line,
            ha="left", va="center", fontsize=8.5, color="#444444")

ax.axhline(1.85, xmin=0.03, xmax=0.97, color="#cccccc", lw=0.8, ls="--")

plt.tight_layout()
plt.savefig(
    "/Users/SubasRW1/Documents/GitHub/ML-Learning-Hub/Ensemble methods/"
    "bagging-and-random-forest-algorithm/lessons/bagging_diagram.png",
    dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
)
print("Saved.")
plt.show()
