import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 4, figsize=(18, 7))
fig.patch.set_facecolor("#FAFAFA")

C_TITLE = "#1a1a2e"
C_TRUE  = "#2ecc71"
C_PRED  = "#3498db"
C_RESID = "#e94560"
C_TREE  = "#533483"
C_NOTE  = "#666666"
C_ARROW = "#555555"
C_BG    = "#f0f4ff"

# Sample data for regression demo
np.random.seed(42)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y_true = np.array([2.5, 4.2, 3.8, 6.1, 5.5, 7.8, 7.2, 9.0])

# Round 0: initial prediction = mean
F0 = np.full_like(y_true, np.mean(y_true))
r0 = y_true - F0   # residuals

# Round 1: tree fits residuals — simplified: threshold split at x<=4
tree1 = np.where(x <= 4, np.mean(r0[x <= 4]), np.mean(r0[x > 4]))
lr = 0.8
F1 = F0 + lr * tree1
r1 = y_true - F1   # residuals after round 1

# Round 2: tree fits r1 — threshold at x<=3
tree2 = np.where(x <= 3, np.mean(r1[x <= 3]), np.mean(r1[x > 3]))
F2 = F1 + lr * tree2
r2 = y_true - F2

def panel_style(ax, title, subtitle=""):
    ax.set_facecolor(C_BG)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color("#cccccc")
    ax.set_xlim(0, 9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"x{i}" for i in x], fontsize=8)
    ax.tick_params(labelsize=8, color="#aaaaaa")
    ax.text(4.5, ax.get_ylim()[1] * 1.02, title, ha="center", fontsize=12,
            fontweight="bold", color=C_TITLE, transform=ax.transData)
    if subtitle:
        ax.text(4.5, ax.get_ylim()[1] * 0.97, subtitle, ha="center", fontsize=8,
                color=C_NOTE, fontstyle="italic", transform=ax.transData)

# ── Panel 0: Data + initial baseline ──────────────────────────────────────────
ax = axes[0]
ax.scatter(x, y_true, color=C_TRUE, s=80, zorder=5, label="True y", clip_on=False)
ax.axhline(np.mean(y_true), color=C_PRED, lw=2.5, ls="--", label=f"F₀ = mean = {np.mean(y_true):.1f}")
for xi, yi, pi in zip(x, y_true, F0):
    ax.plot([xi, xi], [pi, yi], color=C_RESID, lw=1.8, zorder=4)
ax.set_ylim(0, 12)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Step 0 — Initial Prediction\n(baseline = mean of y)", fontsize=10, fontweight="bold", color=C_TITLE, pad=10)
ax.text(4.5, 0.4, "Residuals = vertical red lines", ha="center", fontsize=8, color=C_RESID, fontstyle="italic")

# ── Panel 1: Residuals that Tree 1 fits ───────────────────────────────────────
ax = axes[1]
ax.axhline(0, color="#aaaaaa", lw=1.2, ls="--")
ax.bar(x, r0, color=C_RESID, alpha=0.75, width=0.5, label="Residuals r₀ = y − F₀")
# show tree 1 prediction of residuals
ax.step(np.append(x, 9), np.append(tree1, tree1[-1]), where="post",
        color=C_TREE, lw=2.5, label="Tree₁ fit to r₀")
ax.set_ylim(-3.5, 3.5)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Step 1 — Fit Tree₁ to Residuals\n(pseudo-residuals = errors of F₀)", fontsize=10, fontweight="bold", color=C_TITLE, pad=10)
ax.set_xticks(x)
ax.set_xticklabels([f"x{i}" for i in x], fontsize=8)
ax.text(4.5, -3.0, "Tree learns the direction of the gradient", ha="center", fontsize=8, color=C_TREE, fontstyle="italic")

# ── Panel 2: Updated prediction F1 = F0 + lr*Tree1 ───────────────────────────
ax = axes[2]
ax.scatter(x, y_true, color=C_TRUE, s=80, zorder=5, label="True y")
ax.plot(x, F0, color=C_PRED, lw=1.5, ls=":", alpha=0.6, label="F₀ (old)")
ax.plot(x, F1, color=C_TREE, lw=2.5, ls="--", label=f"F₁ = F₀ + {lr}·Tree₁")
for xi, yi, pi in zip(x, y_true, F1):
    ax.plot([xi, xi], [pi, yi], color=C_RESID, lw=1.8, zorder=4, alpha=0.8)
ax.set_ylim(0, 12)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Step 2 — Update Prediction\nF₁ = F₀ + η · Tree₁", fontsize=10, fontweight="bold", color=C_TITLE, pad=10)
ax.text(4.5, 0.4, f"η (learning rate) = {lr} — shrinks tree contribution", ha="center", fontsize=8, color=C_NOTE, fontstyle="italic")

# ── Panel 3: After round 2 — predictions much closer ─────────────────────────
ax = axes[3]
ax.scatter(x, y_true, color=C_TRUE, s=80, zorder=5, label="True y")
ax.plot(x, F0, color=C_PRED, lw=1.0, ls=":", alpha=0.4, label="F₀ (baseline)")
ax.plot(x, F1, color=C_TREE, lw=1.5, ls=":", alpha=0.5, label="F₁ (round 1)")
ax.plot(x, F2, color="#e67e22", lw=2.5, ls="--", label="F₂ = F₁ + η·Tree₂")
for xi, yi, pi in zip(x, y_true, F2):
    ax.plot([xi, xi], [pi, yi], color=C_RESID, lw=1.8, zorder=4, alpha=0.7)
ax.set_ylim(0, 12)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Step 3 — Round 2: Fit Tree₂ to r₁\nEach round closes the gap", fontsize=10, fontweight="bold", color=C_TITLE, pad=10)
mse0 = np.mean((y_true - F0)**2)
mse2 = np.mean((y_true - F2)**2)
ax.text(4.5, 0.4, f"MSE: {mse0:.2f} → {mse2:.2f} (after 2 rounds)", ha="center", fontsize=8, color="#e67e22", fontweight="bold")

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle("Gradient Boosting — Sequential Residual Fitting",
             fontsize=17, fontweight="bold", color=C_TITLE, y=1.01)

# Connecting arrows between panels
for i in range(3):
    fig.text(0.255 + i * 0.247, 0.5, "→", fontsize=22, color=C_ARROW,
             va="center", ha="center", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(
    "/Users/SubasRW1/Documents/GitHub/ML-Learning-Hub/Ensemble methods/"
    "combining-weak-models-with-boosting/lessons/gradient_boosting_diagram.png",
    dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
)
print("Saved.")
plt.show()
