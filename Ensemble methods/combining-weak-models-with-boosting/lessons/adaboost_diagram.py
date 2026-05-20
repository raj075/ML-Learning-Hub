import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── dataset (from whiteboard) ─────────────────────────────────────────────────
data = [
    [25, 24, 'F'],
    [41, 31, 'F'],
    [56, 28, 'M'],
    [78, 26, 'F'],
    [62, 30, 'M'],
]
N = len(data)

# Stump 1: splits on Age < 50 → Low Risk (0), Age ≥ 50 → High Risk (1)
# True labels (hypothetical: BMI > 28 → High Risk)
true_labels = [0, 1, 0, 0, 1]          # 0=Low, 1=High
stump1_pred  = [0, 0, 0, 1, 1]         # stump1: Age < 53 → Low
w1 = [1/N]*N                           # initial equal weights

# which are wrong on stump1?
wrong1 = [i for i in range(N) if stump1_pred[i] != true_labels[i]]   # idx 1
correct1 = [i for i in range(N) if stump1_pred[i] == true_labels[i]]

err1 = sum(w1[i] for i in wrong1)
alpha1 = 0.5 * np.log((1 - err1) / err1)

# update weights
raw_w2 = []
for i in range(N):
    if i in wrong1:
        raw_w2.append(w1[i] * np.exp(alpha1))
    else:
        raw_w2.append(w1[i] * np.exp(-alpha1))
Z = sum(raw_w2)
w2 = [w / Z for w in raw_w2]

# Stump 2: splits on BMI < 28 → Low, BMI ≥ 28 → High
# BMI values: 24,31,28,26,30 → pred: Low, High, High, Low, High
# sample 2 (BMI=28, true=Low) is misclassified → err2 > 0
stump2_pred = [0, 1, 1, 0, 1]
wrong2   = [i for i in range(N) if stump2_pred[i] != true_labels[i]]
correct2 = [i for i in range(N) if stump2_pred[i] == true_labels[i]]
err2 = sum(w2[i] for i in wrong2)
alpha2 = 0.5 * np.log((1 - err2) / err2)

# ── palette ───────────────────────────────────────────────────────────────────
BG      = '#0f1117'
PANEL   = '#1e2340'
BORDER  = '#2d3561'
ACCENT  = '#5b8af5'
GREEN   = '#34d399'
RED     = '#f87171'
YELLOW  = '#fbbf24'
PURPLE  = '#a78bfa'
TEXT    = '#e2e8f0'
MUTED   = '#94a3b8'
WHITE   = '#ffffff'

fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.text(0.5, 0.97, 'AdaBoost — Step-by-Step', ha='center', va='top',
         fontsize=20, fontweight='bold', color=WHITE, fontfamily='monospace')
fig.text(0.5, 0.935, 'Adaptive Boosting: sequentially train stumps, upweight misclassified samples each round',
         ha='center', va='top', fontsize=10, color=MUTED)

# ── helper functions ──────────────────────────────────────────────────────────
def panel(ax, title, subtitle=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=ACCENT, fontsize=11, fontweight='bold', pad=8)
    if subtitle:
        ax.text(0.5, -0.04, subtitle, ha='center', va='top',
                transform=ax.transAxes, fontsize=8, color=MUTED)

def draw_table(ax, weights, wrong_idx, round_num):
    """Draw the dataset table with weights in a given axes."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    cols = ['Age', 'BMI', 'Gender', 'Label', f'w{round_num}', '']
    col_x = [0.5, 2.1, 3.7, 5.1, 6.5, 8.2]

    # header
    for cx, col in zip(col_x, cols):
        ax.text(cx, 6.4, col, ha='center', va='center',
                fontsize=9, fontweight='bold', color=ACCENT)
    ax.axhline(6.05, color=BORDER, lw=1.2, xmin=0.02, xmax=0.98)

    label_str = ['Low', 'High', 'Low', 'Low', 'High']
    label_col  = [GREEN, RED, GREEN, GREEN, RED]

    for i, (row, lbl, lc, w) in enumerate(zip(data, label_str, label_col, weights)):
        y = 5.1 - i * 1.05
        bg = RED + '22' if i in wrong_idx else PANEL
        rect = FancyBboxPatch((0.1, y - 0.42), 9.8, 0.84,
                              boxstyle='round,pad=0.05',
                              facecolor=bg, edgecolor='none')
        ax.add_patch(rect)

        ax.text(col_x[0], y, str(row[0]), ha='center', va='center', fontsize=9, color=TEXT)
        ax.text(col_x[1], y, str(row[1]), ha='center', va='center', fontsize=9, color=TEXT)
        ax.text(col_x[2], y, row[2],     ha='center', va='center', fontsize=9, color=TEXT)
        ax.text(col_x[3], y, lbl,        ha='center', va='center', fontsize=8.5,
                fontweight='bold', color=lc)
        ax.text(col_x[4], y, f'{w:.3f}', ha='center', va='center', fontsize=9, color=YELLOW)

        if i in wrong_idx:
            ax.text(col_x[5], y, '✗', ha='center', va='center', fontsize=14, color=RED)
        else:
            ax.text(col_x[5], y, '✓', ha='center', va='center', fontsize=14, color=GREEN)

def draw_stump(ax, feature, threshold, direction, alpha, err):
    """Draw a decision stump tree."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)

    # stump node
    rect = FancyBboxPatch((2.5, 5.5), 5, 1.5, boxstyle='round,pad=0.2',
                          facecolor=ACCENT+'33', edgecolor=ACCENT, lw=2)
    ax.add_patch(rect)
    ax.text(5, 6.25, f'{feature}', ha='center', va='center',
            fontsize=10, fontweight='bold', color=WHITE)
    ax.text(5, 5.75, f'threshold: {threshold}', ha='center', va='center',
            fontsize=8.5, color=MUTED)

    # left branch
    ax.annotate('', xy=(2.2, 4.2), xytext=(3.8, 5.5),
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.8, mutation_scale=14))
    ax.text(1.5, 4.95, f'< {threshold}', ha='center', fontsize=8, color=GREEN)
    lbox = FancyBboxPatch((0.4, 3.2), 3.2, 1.0, boxstyle='round,pad=0.15',
                          facecolor=GREEN+'22', edgecolor=GREEN, lw=1.5)
    ax.add_patch(lbox)
    ax.text(2.0, 3.72, '→ Low Risk', ha='center', va='center',
            fontsize=9, fontweight='bold', color=GREEN)

    # right branch
    ax.annotate('', xy=(7.8, 4.2), xytext=(6.2, 5.5),
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.8, mutation_scale=14))
    ax.text(8.5, 4.95, f'≥ {threshold}', ha='center', fontsize=8, color=RED)
    rbox = FancyBboxPatch((6.4, 3.2), 3.2, 1.0, boxstyle='round,pad=0.15',
                          facecolor=RED+'22', edgecolor=RED, lw=1.5)
    ax.add_patch(rbox)
    ax.text(8.0, 3.72, '→ High Risk', ha='center', va='center',
            fontsize=9, fontweight='bold', color=RED)

    # stats box
    sbox = FancyBboxPatch((1.5, 0.4), 7, 2.2, boxstyle='round,pad=0.15',
                          facecolor=PANEL, edgecolor=BORDER, lw=1.5)
    ax.add_patch(sbox)
    ax.text(5, 2.3, f'Error  ε = {err:.3f}', ha='center', va='center',
            fontsize=9.5, color=YELLOW, fontweight='bold')
    ax.text(5, 1.7, f'Weight  α = ½ · ln((1−ε)/ε) = {alpha:.3f}', ha='center', va='center',
            fontsize=9, color=PURPLE)
    hint = 'Higher α → this stump has more say in the final vote'
    ax.text(5, 1.05, hint, ha='center', va='center', fontsize=7.5, color=MUTED, style='italic')

def draw_weight_bar(ax, w1_vals, w2_vals):
    """Visualise old vs new weights as a bar chart."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(1.2)

    x = np.arange(N)
    bars_old = ax.bar(x - 0.22, w1_vals, 0.4, color=ACCENT+'99', label='Before', zorder=3)
    bars_new = ax.bar(x + 0.22, w2_vals, 0.4, color=YELLOW+'cc', label='After',  zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(N)], color=TEXT, fontsize=9)
    ax.set_ylabel('Weight', color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_ylim(0, max(w2_vals) * 1.35)
    ax.axhline(1/N, color=MUTED, lw=0.8, ls='--', zorder=2)
    ax.text(N - 0.5, 1/N + 0.005, 'uniform', color=MUTED, fontsize=7, ha='right')
    ax.set_title('Weight Update After Round 1', color=ACCENT, fontsize=10,
                 fontweight='bold', pad=6)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER,
              labelcolor=TEXT, loc='upper left')
    ax.set_facecolor(PANEL)
    ax.grid(axis='y', color=BORDER, lw=0.6, zorder=1)

    # annotate which sample got upweighted
    for i in wrong1:
        ax.annotate('↑ upweighted\n(misclassified)',
                    xy=(i + 0.22, w2_vals[i]),
                    xytext=(i + 0.9, w2_vals[i] + 0.02),
                    fontsize=7, color=RED,
                    arrowprops=dict(arrowstyle='->', color=RED, lw=1))

# ── layout: 2 rows × 4 cols ───────────────────────────────────────────────────
gs = fig.add_gridspec(2, 4,
                      left=0.03, right=0.97, top=0.9, bottom=0.07,
                      wspace=0.38, hspace=0.55)

ax_d1  = fig.add_subplot(gs[0, 0])   # dataset round 1
ax_s1  = fig.add_subplot(gs[0, 1])   # stump 1
ax_wb  = fig.add_subplot(gs[0, 2])   # weight bar
ax_d2  = fig.add_subplot(gs[0, 3])   # dataset round 2
ax_s2  = fig.add_subplot(gs[1, 0])   # stump 2
ax_fin = fig.add_subplot(gs[1, 1:4]) # final ensemble

# ── PANEL 1: Initial data ─────────────────────────────────────────────────────
panel(ax_d1, 'Round 1 — Initial Dataset', 'All samples get equal weight  w = 1/N')
draw_table(ax_d1, w1, wrong1, round_num=1)

# ── PANEL 2: Stump 1 ─────────────────────────────────────────────────────────
panel(ax_s1, 'Stump 1 — Age < 53', f'Trained on equal weights')
draw_stump(ax_s1, 'Age', 53, '<', alpha1, err1)

# ── PANEL 3: Weight bar ───────────────────────────────────────────────────────
draw_weight_bar(ax_wb, w1, w2)

# ── PANEL 4: Updated data ────────────────────────────────────────────────────
panel(ax_d2, 'Round 2 — Updated Weights', 'Misclassified sample now has higher weight')
draw_table(ax_d2, w2, wrong2, round_num=2)

# ── PANEL 5: Stump 2 ─────────────────────────────────────────────────────────
panel(ax_s2, 'Stump 2 — BMI < 28', 'Trained on re-weighted data')
draw_stump(ax_s2, 'BMI', 28, '<', alpha2, err2)

# ── PANEL 6: Final ensemble ───────────────────────────────────────────────────
ax_fin.set_facecolor(PANEL)
for sp in ax_fin.spines.values():
    sp.set_edgecolor(BORDER); sp.set_linewidth(1.5)
ax_fin.set_xticks([]); ax_fin.set_yticks([])
ax_fin.set_title('Final Ensemble — Weighted Majority Vote', color=ACCENT,
                 fontsize=11, fontweight='bold', pad=8)

ax_fin.set_xlim(0, 14); ax_fin.set_ylim(0, 5)

# stump boxes
for j, (lbl, al, feat, thr) in enumerate([
    (f'Stump 1\nAge < 53', alpha1, 'Age',  53),
    (f'Stump 2\nBMI < 29', alpha2, 'BMI', 29),
]):
    bx = 0.6 + j * 3.8
    rect = FancyBboxPatch((bx, 2.3), 3.0, 2.2, boxstyle='round,pad=0.2',
                          facecolor=ACCENT+'22', edgecolor=ACCENT, lw=1.8)
    ax_fin.add_patch(rect)
    ax_fin.text(bx + 1.5, 3.7, lbl, ha='center', va='center',
                fontsize=9, fontweight='bold', color=WHITE)
    ax_fin.text(bx + 1.5, 3.0, f'α = {al:.3f}', ha='center', va='center',
                fontsize=9, color=PURPLE)
    ax_fin.annotate('', xy=(8.6, 3.4), xytext=(bx + 3.0, 3.4),
                    arrowprops=dict(arrowstyle='-|>', color=ACCENT,
                                   lw=1.5, mutation_scale=12))

# vote box
vbox = FancyBboxPatch((8.6, 1.8), 4.8, 2.8, boxstyle='round,pad=0.2',
                      facecolor=GREEN+'22', edgecolor=GREEN, lw=2)
ax_fin.add_patch(vbox)
ax_fin.text(11.0, 4.1, 'Weighted Vote', ha='center', va='center',
            fontsize=10, fontweight='bold', color=GREEN)
ax_fin.text(11.0, 3.5,
            f'F(x) = sign( α₁·h₁(x) + α₂·h₂(x) + … )',
            ha='center', va='center', fontsize=8.5, color=TEXT, style='italic')
ax_fin.text(11.0, 2.85,
            f'= sign( {alpha1:.2f}·h₁(x) + {alpha2:.2f}·h₂(x) )',
            ha='center', va='center', fontsize=8.5, color=YELLOW)
ax_fin.text(11.0, 2.2,
            'Higher α  →  stump gets more say in the final prediction',
            ha='center', va='center', fontsize=8, color=MUTED, style='italic')

# key insight strip at the bottom
ax_fin.text(7, 1.2,
            'Key idea:  each round re-weights the data so the next stump focuses on what the current ensemble gets wrong.',
            ha='center', va='center', fontsize=9, color=TEXT,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=YELLOW+'22',
                      edgecolor=YELLOW, lw=1.2))
ax_fin.text(7, 0.4,
            'Repeat for k rounds  →  each new stump patches the previous ensemble\'s mistakes  →  bias decreases iteratively',
            ha='center', va='center', fontsize=8, color=MUTED)

out = ("/Users/SubasRW1/Documents/GitHub/ML-Learning-Hub/"
       "Ensemble methods/combining-weak-models-with-boosting/lessons/adaboost_diagram.png")
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved:", out)
plt.show()
