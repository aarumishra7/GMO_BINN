"""
visualize_phase1.py — Generate all Phase 1 diagnostic figures.

Runnable standalone:
    python -m src.visualize_phase1
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from src.utils import load_config, ensure_dirs, load_arrays, gene_labels


# ── colour palette ────────────────────────────────────────────────────────
NATIVE_CMAP  = plt.cm.tab10
TRANS_COLOR  = "#e74c3c"
EDGE_POS     = "#2ecc71"
EDGE_NEG     = "#e74c3c"
PRIOR_ONLY   = "#f39c12"


def plot_adjacency(W, prior, labels, save_path):
    """Side-by-side heatmaps: true adjacency vs imperfect prior."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # True W
    vmax = max(abs(W.min()), abs(W.max()))
    im0 = axes[0].imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[0].set_title("Ground-Truth Adjacency  W", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Regulator  j")
    axes[0].set_ylabel("Target  i")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Weight")

    # Prior (binary)
    im1 = axes[1].imshow(prior, cmap="Oranges", vmin=0, vmax=1, aspect="equal")
    axes[1].set_title("Imperfect Prior  (binary)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Regulator  j")
    axes[1].set_ylabel("Target  i")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlap comparison
    true_bin = (W != 0).astype(float)
    overlap = np.zeros((*W.shape, 3))  # RGB
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            t, p = true_bin[i, j], prior[i, j]
            if t and p:
                overlap[i, j] = [0.18, 0.80, 0.44]   # green  = correctly in prior
            elif t and not p:
                overlap[i, j] = [0.91, 0.30, 0.24]   # red    = true edge missing from prior
            elif not t and p:
                overlap[i, j] = [0.95, 0.61, 0.07]   # orange = spurious prior edge
            else:
                overlap[i, j] = [0.95, 0.95, 0.95]   # light grey = both absent

    axes[2].imshow(overlap, aspect="equal")
    axes[2].set_title("Prior vs Truth  Overlap", fontsize=12, fontweight="bold")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel("Regulator  j")
    axes[2].set_ylabel("Target  i")

    # legend patches
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#2ecc71", label="True edge kept in prior"),
        Patch(facecolor="#e74c3c", label="True edge missing from prior"),
        Patch(facecolor="#f39c12", label="Spurious edge in prior"),
    ]
    axes[2].legend(handles=legend_items, loc="upper left",
                   fontsize=7, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ {save_path}")


def plot_expression_timecourses(t, clean, noisy, labels, n_conditions, save_path):
    """
    Grid of expression trajectories.
    Rows = GMO conditions, Columns = selected genes.
    Solid = clean, dashed = noisy.
    """
    n_genes = clean.shape[2]
    gene_idxs = list(range(min(6, n_genes)))  # show up to 6 genes
    if n_genes - 1 not in gene_idxs:          # always include transgene (last)
        gene_idxs[-1] = n_genes - 1

    n_rows = n_conditions
    n_cols = len(gene_idxs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows),
                             sharex=True, squeeze=False)

    copy_scales = np.linspace(0.5, 2.0, n_conditions)

    for ri in range(n_rows):
        for ci, gi in enumerate(gene_idxs):
            ax = axes[ri][ci]
            color = TRANS_COLOR if gi == n_genes - 1 else NATIVE_CMAP(gi % 10)
            ax.plot(t, clean[ri, :, gi], color=color, lw=1.8, label="Clean")
            ax.plot(t, noisy[ri, :, gi], color=color, lw=0.8, ls="--",
                    alpha=0.5, label="Noisy")
            if ri == 0:
                ax.set_title(labels[gi], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"CN {copy_scales[ri]:.1f}×", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_ylim(bottom=0)
            if ri == n_rows - 1:
                ax.set_xlabel("t", fontsize=8)

    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Expression Time-Courses  (rows = GMO lines, columns = genes)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ {save_path}")


def plot_transgene_impact(W, labels, save_path):
    """
    Bar chart showing the transgene's regulatory weights on native genes.
    """
    n = W.shape[0]
    trans_col = W[:, -1]   # last column = transgene as regulator
    native_labels = labels[:-1]
    weights = trans_col[:-1]

    colours = [EDGE_POS if w > 0 else EDGE_NEG if w < 0 else "#cccccc"
               for w in weights]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(weights)), weights, color=colours, edgecolor="white", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(native_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Regulatory Weight  W[target, transgene]", fontsize=10)
    ax.set_title("Transgene → Native Gene Regulatory Impact  (Ground Truth)",
                 fontsize=12, fontweight="bold")

    for i, (bar, w) in enumerate(zip(bars, weights)):
        if w != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, w,
                    f"{w:+.2f}", ha="center",
                    va="bottom" if w > 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ {save_path}")


def plot_train_test_split(t_train, t_test, x_train, x_test, labels, save_path):
    """
    Show the temporal train/test split for condition 0, all genes overlaid.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    n_genes = x_train.shape[2]

    for gi in range(n_genes):
        color = TRANS_COLOR if gi == n_genes - 1 else NATIVE_CMAP(gi % 10)
        lw = 2.0 if gi == n_genes - 1 else 1.0
        ax.plot(t_train, x_train[0, :, gi], color=color, lw=lw)
        ax.plot(t_test, x_test[0, :, gi], color=color, lw=lw, ls="--", alpha=0.5)

    ax.axvline(t_train[-1], color="grey", ls=":", lw=1.5, label="Train | Test boundary")
    ax.fill_betweenx([0, ax.get_ylim()[1] * 1.05],
                     t_test[0], t_test[-1], alpha=0.07, color="red")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Expression (noisy)", fontsize=11)
    ax.set_title("Train / Test Split  (condition 0, all genes — dashed = held-out extrapolation)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_visualizations(cfg=None):
    if cfg is None:
        cfg = load_config()
    ensure_dirs("figures")

    print("\n  Generating Phase 1 figures...")

    arrays = load_arrays("data",
                         "grn_adjacency_true", "prior_adjacency",
                         "expression_clean", "expression_noisy",
                         "time_points", "t_train", "t_test",
                         "x_train", "x_test")

    labels = np.array(gene_labels(cfg["grn"]["n_native_genes"],
                                  cfg["grn"]["n_transgenes"]))

    plot_adjacency(arrays["grn_adjacency_true"],
                   arrays["prior_adjacency"],
                   labels,
                   "figures/01_adjacency_and_prior.png")

    plot_expression_timecourses(
        arrays["time_points"],
        arrays["expression_clean"],
        arrays["expression_noisy"],
        labels,
        cfg["data"]["n_conditions"],
        "figures/02_expression_timecourses.png")

    plot_transgene_impact(arrays["grn_adjacency_true"],
                          labels,
                          "figures/03_transgene_impact.png")

    plot_train_test_split(arrays["t_train"], arrays["t_test"],
                          arrays["x_train"], arrays["x_test"],
                          labels,
                          "figures/04_train_test_split.png")

    print("\n  ✓  All Phase 1 figures saved to figures/\n")


if __name__ == "__main__":
    run_visualizations()