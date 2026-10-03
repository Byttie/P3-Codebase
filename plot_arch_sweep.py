"""
plot_arch_sweep.py
------------------
Plots the RoutingGuard architecture-sweep results for section 5.2
("Analysis of Design Solutions").

The figure shows held-out cross-validated F1 as the GRU is scaled up along
two axes -- number of layers (1, 2, 3) and hidden size (128, 512, 2048) --
for each of the three models. The message the figure is meant to carry is
that F1 stays essentially flat as the model grows, i.e. scaling the GRU does
not help and the lean 1-layer / 128-hidden detector is the right choice.

Only pandas + matplotlib are required (no seaborn, no sklearn).

Run:
    python plot_arch_sweep.py

Outputs (written next to this script):
    arch_sweep.png   (300 dpi, for Word)
    arch_sweep.pdf   (vector, for LaTeX)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "arch_sweep.csv")
df = pd.read_csv(CSV)

# Fixed model order (left -> right) and the hidden sizes on the x-axis.
MODEL_ORDER = ["ZAYA-1 8B", "LFM2.5", "Phi-mini-MoE"]
HIDDEN = [128, 512, 2048]
LAYERS = [1, 2, 3]

# One colour per layer-count. Colour-blind-safe, prints fine in greyscale
# because the three also get different markers.
LAYER_STYLE = {
    1: {"color": "#0072B2", "marker": "o", "label": "1 layer"},
    2: {"color": "#D55E00", "marker": "s", "label": "2 layers"},
    3: {"color": "#009E73", "marker": "^", "label": "3 layers"},
}

# ----------------------------------------------------------------------
# 2. Figure: one panel per model, shared y-axis
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)

x = list(range(len(HIDDEN)))            # even spacing: 0,1,2 for 128,512,2048

for ax, model in zip(axes, MODEL_ORDER):
    sub = df[df["model"] == model]
    kval = int(sub["topk_K"].iloc[0])   # pooling K used for this model

    for layers in LAYERS:
        s = sub[sub["layers"] == layers].sort_values("hidden")
        st = LAYER_STYLE[layers]
        ax.errorbar(
            x, s["f1_mean"].values,
            yerr=s["f1_std"].values,
            color=st["color"], marker=st["marker"], markersize=6,
            linewidth=1.8, capsize=3, elinewidth=1.2,
            label=st["label"],
        )

    # Highlight the lean, reported configuration (1 layer, 128 hidden).
    lean = sub[(sub["layers"] == 1) & (sub["hidden"] == 128)]
    ax.scatter([0], lean["f1_mean"].values, s=170, facecolors="none",
               edgecolors="black", linewidths=1.6, zorder=5)
    ax.annotate("reported\n(1x128)", xy=(0, lean["f1_mean"].iloc[0]),
                xytext=(0.05, 0.12), textcoords="axes fraction",
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_title(f"{model}\n(top-K K={kval}, z-score on)")
    ax.set_xticks(x)
    ax.set_xticklabels(HIDDEN)
    ax.set_xlabel("GRU hidden size")
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)
    ax.xaxis.set_minor_formatter(NullFormatter())

axes[0].set_ylabel("Cross-validated F1 (mean +/- std)")

# Shared y-limits chosen so the flatness within each model AND the level
# differences between models are both visible.
axes[0].set_ylim(0.88, 0.965)

# Single shared legend above the panels.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("RoutingGuard architecture sweep",
             y=1.10, fontsize=13)

fig.tight_layout()

# ----------------------------------------------------------------------
# 3. Save
# ----------------------------------------------------------------------
png = os.path.join(HERE, "arch_sweep.png")
pdf = os.path.join(HERE, "arch_sweep.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print("wrote:", png)
print("wrote:", pdf)
