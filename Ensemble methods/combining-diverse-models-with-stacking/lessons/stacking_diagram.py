import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

C_TITLE  = "#1a1a2e"
C_DATA   = "#16213e"
C_L1     = "#0f3460"
C_META   = "#e94560"
C_ARROW  = "#555555"
C_OUT    = "#2ecc71"
C_NOTE   = "#777777"
C_PRED   = "#533483"
C_SPLIT  = "#e67e22"

def rounded_box(ax, xy, w, h, color, alpha=0.12, lw=2, ls="-"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         linewidth=lw, linestyle=ls,
                         edgecolor=color,
                         facecolor=color, alpha=alpha)
    ax.add_patch(box)

def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13))

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax.text(7.5, 8.6, "Stacking  ·  Ensemble with a Meta-Learner",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color=C_TITLE, fontfamily="monospace")
ax.axhline(8.35, xmin=0.03, xmax=0.97, color=C_TITLE, lw=1.2)

# ── TRAINING DATA ─────────────────────────────────────────────────────────────
rounded_box(ax, (0.3, 3.6), 2.0, 3.8, C_DATA, alpha=0.10, lw=2.5)
ax.text(1.3, 7.1, "Training", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_DATA)
ax.text(1.3, 6.65, "Data", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_DATA)
ax.text(1.3, 6.15, "X_train", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_DATA, fontstyle="italic")
ax.text(1.3, 5.6, "y_train", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_DATA, fontstyle="italic")

# Train / Val split indicator
ax.axhline(5.0, xmin=0.032, xmax=0.162, color=C_SPLIT, lw=1.5, ls="--")
ax.text(1.3, 4.65, "Train", ha="center", fontsize=8.5, color=C_SPLIT, fontweight="bold")
ax.text(1.3, 4.25, "Fold k", ha="center", fontsize=8, color=C_SPLIT)
ax.text(1.3, 3.85, "Val Fold k", ha="center", fontsize=8, color=C_SPLIT)

# ── LEVEL-1 BASE MODELS ───────────────────────────────────────────────────────
base_models = [
    (7.1, "KNN", "K-Nearest\nNeighbour", C_L1),
    (5.0, "RF",  "Random\nForest",       "#533483"),
    (2.9, "NB",  "Naive\nBayes",         "#0a7c59"),
]

for (yc, tag, label, color) in base_models:
    bx, by = 3.8, yc - 0.55
    rounded_box(ax, (bx, by), 2.2, 1.1, color, alpha=0.13, lw=1.8)
    ax.text(bx + 1.1, yc + 0.05, tag,
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=color)
    ax.text(bx + 1.1, yc - 0.3, label,
            ha="center", va="center", fontsize=8.5, color=color)
    # arrow: data → model
    arrow(ax, 2.3, yc - 0.0, bx + 0.05, yc - 0.0, color=color, lw=1.4)

# ── LEVEL-1 PREDICTIONS (stacked matrix) ─────────────────────────────────────
px, py = 7.0, 3.9
rounded_box(ax, (px, py), 2.4, 3.5, C_PRED, alpha=0.10, lw=2.0)
ax.text(px + 1.2, py + 3.15, "Predictions", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_PRED)
ax.text(px + 1.2, py + 2.65, "(new features)", ha="center", va="center",
        fontsize=8.5, color=C_PRED, fontstyle="italic")

# table header
ax.text(px + 0.55, py + 2.1, "KNN", ha="center", fontsize=8, color=C_L1, fontweight="bold")
ax.text(px + 1.2,  py + 2.1, "RF",  ha="center", fontsize=8, color="#533483", fontweight="bold")
ax.text(px + 1.9,  py + 2.1, "NB",  ha="center", fontsize=8, color="#0a7c59", fontweight="bold")
ax.axhline(py + 1.95, xmin=px/15, xmax=(px+2.4)/15, color=C_PRED, lw=0.8, ls="--")

sample_preds = [("0","1","0"), ("1","1","1"), ("0","0","1"), ("1","1","0")]
cols = [px+0.55, px+1.2, px+1.9]
for row_i, (p1, p2, p3) in enumerate(sample_preds):
    yrow = py + 1.6 - row_i * 0.38
    for ci, val in enumerate([p1, p2, p3]):
        color = C_OUT if val == "1" else C_META
        ax.text(cols[ci], yrow, val, ha="center", fontsize=9,
                fontweight="bold", color=color)

# arrows: models → predictions matrix
for (yc, _, _, color) in base_models:
    arrow(ax, 6.0, yc, px + 0.05, 5.65, color=C_PRED, lw=1.1)

# ── META-LEARNER ──────────────────────────────────────────────────────────────
mx, my = 10.2, 4.7
rounded_box(ax, (mx, my), 2.6, 2.0, C_META, alpha=0.12, lw=2.2)
ax.text(mx + 1.3, my + 1.7, "Meta-Learner", ha="center", va="center",
        fontsize=10, fontweight="bold", color=C_META)
ax.text(mx + 1.3, my + 1.2, "Logistic\nRegression", ha="center", va="center",
        fontsize=9, color=C_META)
ax.text(mx + 1.3, my + 0.55, "Learns to weight\nand blend L1\npredictions",
        ha="center", va="center", fontsize=7.5, color=C_META, fontstyle="italic")

# arrow: predictions → meta-learner
arrow(ax, px + 2.4, 5.65, mx + 0.05, 5.7, color=C_META, lw=1.6)

# ── FINAL OUTPUT ──────────────────────────────────────────────────────────────
arrow(ax, mx + 2.6, 5.7, 13.8, 5.7, color=C_OUT, lw=2.0)
rounded_box(ax, (13.8, 5.2), 1.0, 1.0, C_OUT, alpha=0.15, lw=2.0)
ax.text(14.3, 5.7, "1", ha="center", va="center",
        fontsize=22, fontweight="bold", color=C_OUT)
ax.text(14.3, 5.2, "Final\nPred", ha="center", va="center",
        fontsize=7.5, color=C_NOTE)

# ── LEVEL LABELS ──────────────────────────────────────────────────────────────
ax.text(1.3, 7.85, "Level 0", ha="center", fontsize=9,
        fontweight="bold", color=C_DATA,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe", edgecolor=C_DATA, lw=1))
ax.text(4.9, 7.85, "Level 1  (Base Models)", ha="center", fontsize=9,
        fontweight="bold", color=C_L1,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe", edgecolor=C_L1, lw=1))
ax.text(8.2, 7.85, "L1 Predictions", ha="center", fontsize=9,
        fontweight="bold", color=C_PRED,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e8ff", edgecolor=C_PRED, lw=1))
ax.text(11.5, 7.85, "Level 2  (Meta-Learner)", ha="center", fontsize=9,
        fontweight="bold", color=C_META,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fde8ed", edgecolor=C_META, lw=1))

# ── ACCURACY RESULTS ─────────────────────────────────────────────────────────
results = [
    ("KNN",              "91.3%", C_L1),
    ("Random Forest",    "93.4%", "#533483"),
    ("Naive Bayes",      "92.1%", "#0a7c59"),
    ("Stacking (all)",   "94.7%", C_META),
]
rx = 0.5
ry = 3.1
ax.text(rx, ry, "3-fold CV Accuracy:", ha="left", fontsize=8.5,
        fontweight="bold", color=C_NOTE)
for i, (name, acc, color) in enumerate(results):
    y_pos = ry - 0.38 - i * 0.38
    ax.text(rx, y_pos, f"{name}", ha="left", fontsize=8, color=color)
    weight = "bold" if name == "Stacking (all)" else "normal"
    ax.text(rx + 3.2, y_pos, acc, ha="right", fontsize=8,
            fontweight=weight, color=color)

ax.axhline(1.8, xmin=0.03, xmax=0.97, color="#cccccc", lw=0.8, ls="--")

# ── KEY NOTES ─────────────────────────────────────────────────────────────────
notes = [
    r"$\bf{Key\ idea}$: train diverse models, use their predictions as features for a meta-learner",
    r"$\bf{Meta-learner}$: learns which base models to trust and in which situations",
    r"$\bf{Diversity\ matters}$: use different model families (KNN, RF, NB) not copies of one model",
    r"$\bf{CV\ scheme}$: base models trained on fold k, predict on held-out fold to avoid leakage",
]
for i, line in enumerate(notes):
    ax.text(0.5, 1.5 - i * 0.37, line,
            ha="left", va="center", fontsize=8.5, color="#444444")

plt.tight_layout()
plt.savefig(
    "/Users/SubasRW1/Documents/GitHub/ML-Learning-Hub/Ensemble methods/"
    "combining-diverse-models-with-stacking/lessons/stacking_diagram.png",
    dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
)
print("Saved.")
plt.show()
