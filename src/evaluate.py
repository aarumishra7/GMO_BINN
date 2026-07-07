"""
evaluate.py — Phase 3: Evaluate & compare the Vanilla Neural ODE and the BINN.

Runnable standalone:
    python -m src.evaluate

Three evaluations, matching the article's Box 1 / Section 5:

  1. Predictive accuracy  — integrate each trained model forward from the
                             training initial condition across the full
                             time range, then score the held-out
                             extrapolation window (t > t_train[-1]) against
                             both the noisy measurements and the
                             noise-free ground truth.

  2. Network recovery     — AUROC / AUPR of each model's learned
                             interaction matrix against the true adjacency,
                             plus a case-study-specific check: does the
                             model's top-K predicted transgene targets
                             match the true transgene targets?

  3. Identifiability      — retrain each architecture from several random
                             seeds and check whether the recovered
                             transgene-target edges are stable. This is the
                             expensive step (see config.yaml's
                             evaluation.stress_test_epochs to control the
                             per-seed training budget).

Evaluations 1 and 2 use the single-seed models already trained by
`python -m src.train` (loaded from outputs/*.pt). Evaluation 3 retrains
fresh models from scratch, one per seed, per architecture.
"""

import csv
import time
from typing import cast

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, r2_score)
from torchdiffeq import odeint

from src.utils import load_config, seed_everything, ensure_dirs, load_arrays, gene_labels
from src.model_vanilla import VanillaODE
from src.model_binn import BINN_ODE
from src.train import load_phase1_data, train_one_model, extract_interactions


# ═══════════════════════════════════════════════════════════════════════════
# Small numeric helpers
# ═══════════════════════════════════════════════════════════════════════════

def _flat(x) -> np.ndarray:
    """Flatten a torch.Tensor or np.ndarray to a 1-D numpy array."""
    if torch.is_tensor(x):
        return x.detach().numpy().ravel()
    return np.asarray(x).ravel()


def _mse(pred, target) -> float:
    return float(np.mean((_flat(pred) - _flat(target)) ** 2))


def _r2(pred, target) -> float:
    """sklearn's r2_score wants (y_true, y_pred) — target comes first."""
    return float(r2_score(_flat(target), _flat(pred)))


# ═══════════════════════════════════════════════════════════════════════════
# Load the single-seed models trained in Phase 2
# ═══════════════════════════════════════════════════════════════════════════

def load_trained_models(cfg: dict, data: dict) -> tuple[VanillaODE, BINN_ODE]:
    """Reconstruct both architectures and load their Phase 2 checkpoints."""
    n_genes    = cfg["grn"]["n_total"]
    hidden_dim = cfg["model"]["hidden_dim"]
    n_layers   = cfg["model"]["n_hidden_layers"]

    vanilla = VanillaODE(n_genes, hidden_dim, n_layers)
    vanilla.load_state_dict(
        torch.load("outputs/vanilla_ode_model.pt", map_location="cpu", weights_only=True)
    )
    vanilla.eval()

    binn = BINN_ODE(n_genes, hidden_dim, data["prior"],
                    hill_n=cfg["dynamics"]["hill_n"],
                    hill_k=cfg["dynamics"]["hill_k"])
    binn.load_state_dict(
        torch.load("outputs/binn_model.pt", map_location="cpu", weights_only=True)
    )
    binn.eval()

    return vanilla, binn


# ═══════════════════════════════════════════════════════════════════════════
# 1. Predictive accuracy — extrapolation
# ═══════════════════════════════════════════════════════════════════════════

def integrate_full_trajectory(model, data: dict, t_full: torch.Tensor) -> torch.Tensor:
    """Integrate the model forward across the full time range from the
    training initial condition. Returns shape (T_total, n_conditions, n_genes)."""
    y0 = data["x_train"][:, 0, :]
    model.eval()
    with torch.no_grad():
        pred = odeint(model, y0, t_full, method="dopri5",
                      rtol=1e-5, atol=1e-6,
                      options={"max_num_steps": 2000})
    # See the matching note in train.py: torchdiffeq's stubs don't pin down
    # odeint's return type, so Pylance sees it as Unknown. This cast is a
    # no-op at runtime that just tells the checker what we already know.
    pred = cast(torch.Tensor, pred)
    return pred


def evaluate_extrapolation(model, data: dict, model_name: str,
                          verbose: bool = True) -> dict:
    """
    Score a trained model's forward-integrated trajectory on the held-out
    extrapolation window, against both the noisy training-style measurements
    and (if available) the noise-free ground truth.
    """
    t_full  = torch.cat([data["t_train"], data["t_test"]])
    n_train = data["t_train"].shape[0]

    pred_full = integrate_full_trajectory(model, data, t_full)
    pred_full_np = pred_full.numpy()

    pred_train = pred_full[:n_train]
    pred_test  = pred_full[n_train:]

    tgt_train_noisy = data["x_train"].permute(1, 0, 2)
    tgt_test_noisy  = data["x_test"].permute(1, 0, 2)

    mse_train = _mse(pred_train, tgt_train_noisy)
    mse_test  = _mse(pred_test,  tgt_test_noisy)
    r2_train  = _r2(pred_train, tgt_train_noisy)
    r2_test   = _r2(pred_test,  tgt_test_noisy)

    # secondary metric: score against the noise-free ground truth, since
    # we have privileged access to it in this synthetic setup
    clean_metrics: dict = {}
    try:
        clean_full = load_arrays("data", "expression_clean")["expression_clean"]
        clean_test = torch.tensor(clean_full[:, n_train:, :],
                                  dtype=torch.float32).permute(1, 0, 2)
        clean_metrics = {
            "mse_test_clean": _mse(pred_test, clean_test),
            "r2_test_clean":  _r2(pred_test, clean_test),
        }
    except FileNotFoundError:
        pass

    if verbose:
        print(f"\n  [{model_name}]  Extrapolation Evaluation")
        print(f"      Train  MSE = {mse_train:.6f}   R²  = {r2_train:7.4f}")
        print(f"      Test   MSE = {mse_test:.6f}   R²  = {r2_test:7.4f}   "
              f"(vs noisy measurements)")
        if clean_metrics:
            print(f"      Test   MSE = {clean_metrics['mse_test_clean']:.6f}   "
                  f"R²  = {clean_metrics['r2_test_clean']:7.4f}   "
                  f"(vs noise-free ground truth)")

    return {
        "mse_train": mse_train, "r2_train": r2_train,
        "mse_test": mse_test, "r2_test": r2_test,
        "pred_full": pred_full_np,
        **clean_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Network recovery
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_network_recovery(W_pred: np.ndarray, W_true: np.ndarray,
                              model_name: str, verbose: bool = True) -> dict:
    """
    Score a learned interaction matrix against the true adjacency using
    AUROC and AUPR on the off-diagonal binary edge-presence labels.
    """
    n = W_true.shape[0]
    true_bin = (W_true != 0).astype(int)
    np.fill_diagonal(true_bin, 0)
    mask = ~np.eye(n, dtype=bool)

    y_true  = true_bin[mask]
    y_score = np.abs(W_pred)[mask]

    auroc = roc_auc_score(y_true, y_score)
    aupr  = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    baseline_aupr = float(y_true.mean())   # AUPR of a random classifier = positive rate

    if verbose:
        print(f"\n  [{model_name}]  Network Recovery")
        print(f"      AUROC = {auroc:.4f}   (random baseline = 0.500)")
        print(f"      AUPR  = {aupr:.4f}   (random baseline = {baseline_aupr:.4f})")

    return {
        "auroc": auroc, "aupr": aupr, "baseline_aupr": baseline_aupr,
        "fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
    }


def evaluate_transgene_targets(W_pred: np.ndarray, W_true: np.ndarray,
                               transgene_idx: int, model_name: str,
                               k: int = 3, verbose: bool = True) -> dict:
    """
    Case-study-specific metric: of all native genes, which does the model
    say the transgene regulates, and does that match the ground truth?
    """
    true_col = W_true[:, transgene_idx].copy()
    true_col[transgene_idx] = 0
    pred_col = np.abs(W_pred[:, transgene_idx]).copy()
    pred_col[transgene_idx] = 0

    true_targets = set(np.where(true_col != 0)[0].tolist())
    top_k_idx = np.argsort(-pred_col)[:k].tolist()
    top_k_set = set(top_k_idx)

    hits = true_targets & top_k_set
    recall_at_k = len(hits) / max(len(true_targets), 1)

    if verbose:
        print(f"\n  [{model_name}]  Transgene Target Recovery  (top-{k})")
        print(f"      True transgene targets   : {sorted(true_targets)}")
        print(f"      Top-{k} predicted targets : {top_k_idx}")
        print(f"      Recall@{k}                : {recall_at_k:.2f}  "
              f"({len(hits)}/{max(len(true_targets), 1)} true targets recovered)")

    return {
        "true_targets": true_targets, "top_k_idx": top_k_idx,
        "recall_at_k": recall_at_k,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Identifiability stress test
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_seed(model_type: str, cfg: dict, data: dict,
                   n_seeds: int, n_epochs_override: int,
                   verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrain a fresh model from `n_seeds` different random initializations
    and collect each seed's learned interaction matrix + network-recovery
    AUROC.
    """
    matrices = []
    aurocs = []
    n_genes = cfg["grn"]["n_total"]

    print(f"\n  Retraining {model_type} from {n_seeds} seeds "
          f"({n_epochs_override} epochs each)...")

    for seed in range(n_seeds):
        t0 = time.time()
        seed_everything(seed)

        if model_type == "vanilla":
            model = VanillaODE(n_genes, cfg["model"]["hidden_dim"],
                              cfg["model"]["n_hidden_layers"])
            is_binn = False
        else:
            model = BINN_ODE(n_genes, cfg["model"]["hidden_dim"], data["prior"],
                             hill_n=cfg["dynamics"]["hill_n"],
                             hill_k=cfg["dynamics"]["hill_k"])
            is_binn = True

        train_one_model(model, data, cfg, f"{model_type}-seed{seed}",
                        is_binn=is_binn, verbose=verbose,
                        n_epochs_override=n_epochs_override)

        W = extract_interactions(model, data, f"{model_type}-seed{seed}",
                                 is_binn, verbose=False)
        rec = evaluate_network_recovery(W, data["W_true"],
                                        f"{model_type}-seed{seed}", verbose=False)

        matrices.append(W)
        aurocs.append(rec["auroc"])
        print(f"      seed {seed}: AUROC={rec['auroc']:.4f}  "
              f"({time.time() - t0:.1f}s)")

    return np.stack(matrices), np.array(aurocs)


def pairwise_stability(matrices: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Mean pairwise Pearson correlation between the flattened off-diagonal
    interaction weights across seeds. Higher = more stable/identifiable.
    """
    n_seeds, n, _ = matrices.shape
    mask = ~np.eye(n, dtype=bool)
    flat = np.stack([m[mask] for m in matrices])   # (n_seeds, n*(n-1))
    corr = np.corrcoef(flat)                        # (n_seeds, n_seeds)
    iu = np.triu_indices(n_seeds, k=1)
    return corr, float(corr[iu].mean())


def transgene_topk_jaccard(matrices: np.ndarray, transgene_idx: int,
                           k: int = 3) -> tuple[float, list]:
    """
    Mean pairwise Jaccard overlap of each seed's top-K predicted
    transgene-target genes. 1.0 = every seed agrees; 0.0 = no overlap.
    """
    n_seeds = matrices.shape[0]
    top_sets = []
    for m in matrices:
        col = np.abs(m[:, transgene_idx]).copy()
        col[transgene_idx] = 0
        top_sets.append(set(np.argsort(-col)[:k].tolist()))

    jaccards = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccards.append(inter / union if union > 0 else 0.0)

    return float(np.mean(jaccards)), top_sets


def run_identifiability_stress_test(cfg: dict, data: dict,
                                    verbose: bool = True) -> dict:
    """
    For each architecture: retrain from multiple seeds, then measure
    (a) spread of network-recovery AUROC across seeds,
    (b) mean pairwise correlation of the full recovered interaction matrix,
    (c) Jaccard stability of the top-K transgene-target prediction.
    """
    n_seeds           = cfg["evaluation"]["n_seeds"]
    n_epochs_override = cfg["evaluation"]["stress_test_epochs"]
    k                 = cfg["evaluation"]["top_k_transgene"]
    transgene_idx     = cfg["grn"]["n_native_genes"]

    results = {}
    for model_type in ("vanilla", "binn"):
        matrices, aurocs = run_multi_seed(
            model_type, cfg, data, n_seeds, n_epochs_override, verbose=False
        )
        corr, mean_corr = pairwise_stability(matrices)
        jaccard, top_sets = transgene_topk_jaccard(matrices, transgene_idx, k=k)

        if verbose:
            print(f"\n  [{model_type}]  Identifiability Summary  (n={n_seeds} seeds)")
            print(f"      AUROC across seeds        : "
                  f"{aurocs.mean():.4f} ± {aurocs.std():.4f}")
            print(f"      Mean pairwise correlation  : {mean_corr:.4f}   "
                  f"(1.0 = perfectly stable)")
            print(f"      Transgene top-{k} Jaccard    : {jaccard:.4f}   "
                  f"(1.0 = every seed agrees)")

        results[model_type] = {
            "matrices": matrices, "aurocs": aurocs,
            "corr_matrix": corr, "mean_corr": mean_corr,
            "jaccard": jaccard, "top_sets": top_sets,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

NATIVE_CMAP   = plt.cm.tab10
TRUE_COLOR    = "#2c3e50"
VANILLA_COLOR = "#3498db"
BINN_COLOR    = "#e74c3c"


def plot_extrapolation_comparison(vanilla_result: dict, binn_result: dict,
                                  data: dict, labels: np.ndarray,
                                  save_path: str) -> None:
    """
    For a representative condition, overlay Vanilla and BINN predicted
    trajectories against the true (noisy) data, marking the train/test
    boundary — the extrapolation gap both models must bridge.
    """
    t_train = data["t_train"].numpy()
    t_test  = data["t_test"].numpy()
    t_full  = np.concatenate([t_train, t_test])
    n_genes = data["x_train"].shape[2]

    cond = 0  # representative condition
    gene_idxs = list(range(min(6, n_genes)))
    if n_genes - 1 not in gene_idxs:
        gene_idxs[-1] = n_genes - 1

    true_full = np.concatenate(
        [data["x_train"][cond].numpy(), data["x_test"][cond].numpy()], axis=0
    )
    pred_v = vanilla_result["pred_full"][:, cond, :]
    pred_b = binn_result["pred_full"][:, cond, :]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    axes = axes.ravel()

    for ax, gi in zip(axes, gene_idxs):
        ax.plot(t_full, true_full[:, gi], color=TRUE_COLOR, lw=1.8,
                label="True (noisy)")
        ax.plot(t_full, pred_v[:, gi], color=VANILLA_COLOR, lw=1.6, ls="--",
                label="Vanilla pred")
        ax.plot(t_full, pred_b[:, gi], color=BINN_COLOR, lw=1.6, ls="--",
                label="BINN pred")
        ax.axvline(t_train[-1], color="grey", ls=":", lw=1.2)
        ax.set_title(str(labels[gi]), fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Extrapolation: Predicted vs. True Expression  "
        "(condition 0, dotted line = train | test boundary)",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ {save_path}")


def plot_roc_pr_curves(vanilla_rec: dict, binn_rec: dict, save_path: str) -> None:
    """Side-by-side ROC and Precision-Recall curves for network recovery."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(vanilla_rec["fpr"], vanilla_rec["tpr"], color=VANILLA_COLOR, lw=2,
                label=f"Vanilla (AUROC={vanilla_rec['auroc']:.3f})")
    axes[0].plot(binn_rec["fpr"], binn_rec["tpr"], color=BINN_COLOR, lw=2,
                label=f"BINN (AUROC={binn_rec['auroc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC — Network Recovery", fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].plot(vanilla_rec["recall"], vanilla_rec["precision"], color=VANILLA_COLOR,
                lw=2, label=f"Vanilla (AUPR={vanilla_rec['aupr']:.3f})")
    axes[1].plot(binn_rec["recall"], binn_rec["precision"], color=BINN_COLOR,
                lw=2, label=f"BINN (AUPR={binn_rec['aupr']:.3f})")
    axes[1].axhline(vanilla_rec["baseline_aupr"], color="grey", ls="--", lw=1,
                    label="Random baseline")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall — Network Recovery", fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ {save_path}")


def plot_identifiability(id_results: dict, cfg: dict, save_path: str) -> None:
    """
    Four-panel identifiability summary: AUROC spread across seeds, the
    two models' seed-pairwise correlation heatmaps, and transgene-target
    Jaccard stability.
    """
    fig = plt.figure(figsize=(16, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1, 1, 1])

    # (a) AUROC spread across seeds
    ax0 = fig.add_subplot(gs[0])
    box_data = [id_results["vanilla"]["aurocs"], id_results["binn"]["aurocs"]]
    bp = ax0.boxplot(box_data, tick_labels=["Vanilla", "BINN"],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [VANILLA_COLOR, BINN_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax0.axhline(0.5, color="grey", ls="--", lw=1)
    ax0.set_ylabel("AUROC (network recovery)")
    ax0.set_title("AUROC spread\nacross seeds", fontsize=10, fontweight="bold")

    # (b) Vanilla pairwise correlation heatmap
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(id_results["vanilla"]["corr_matrix"], cmap="RdBu_r",
                     vmin=-1, vmax=1)
    ax1.set_title(f"Vanilla: seed-pair corr.\nmean={id_results['vanilla']['mean_corr']:.3f}",
                 fontsize=9)
    ax1.set_xlabel("seed")
    ax1.set_ylabel("seed")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    # (c) BINN pairwise correlation heatmap
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(id_results["binn"]["corr_matrix"], cmap="RdBu_r",
                     vmin=-1, vmax=1)
    ax2.set_title(f"BINN: seed-pair corr.\nmean={id_results['binn']['mean_corr']:.3f}",
                 fontsize=9)
    ax2.set_xlabel("seed")
    ax2.set_ylabel("seed")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    # (d) transgene top-k Jaccard stability
    ax3 = fig.add_subplot(gs[3])
    jvals = [id_results["vanilla"]["jaccard"], id_results["binn"]["jaccard"]]
    ax3.bar(["Vanilla", "BINN"], jvals, color=[VANILLA_COLOR, BINN_COLOR], alpha=0.7)
    ax3.set_ylim(0, 1)
    k = cfg["evaluation"]["top_k_transgene"]
    ax3.set_title(f"Transgene top-{k} target\nJaccard stability", fontsize=9,
                 fontweight="bold")
    ax3.set_ylabel("Mean pairwise Jaccard")

    fig.suptitle("Identifiability Stress Test — Stability Across Random Seeds",
                fontsize=13, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_phase3(cfg: dict | None = None, verbose: bool = True,
              skip_stress_test: bool = False) -> dict:
    """Run all three Phase 3 evaluations and save results + figures."""
    if cfg is None:
        cfg = load_config()

    ensure_dirs("outputs", "figures")

    if verbose:
        print("=" * 60)
        print("  PHASE 3 — Evaluation")
        print("=" * 60)

    data = load_phase1_data()
    labels = np.array(gene_labels(cfg["grn"]["n_native_genes"],
                                  cfg["grn"]["n_transgenes"]))
    transgene_idx = cfg["grn"]["n_native_genes"]

    vanilla, binn = load_trained_models(cfg, data)

    # ── 1. Extrapolation ──────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 60)
        print("  [1/3]  Predictive Accuracy — Extrapolation")
        print("─" * 60)
    vanilla_extra = evaluate_extrapolation(vanilla, data, "Vanilla", verbose=verbose)
    binn_extra    = evaluate_extrapolation(binn, data, "BINN", verbose=verbose)

    # ── 2. Network recovery ──────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 60)
        print("  [2/3]  Network Recovery")
        print("─" * 60)
    W_vanilla = load_arrays("outputs", "interaction_vanilla")["interaction_vanilla"]
    W_binn    = load_arrays("outputs", "interaction_binn")["interaction_binn"]

    vanilla_rec = evaluate_network_recovery(W_vanilla, data["W_true"], "Vanilla",
                                            verbose=verbose)
    binn_rec    = evaluate_network_recovery(W_binn, data["W_true"], "BINN",
                                            verbose=verbose)

    vanilla_tg = evaluate_transgene_targets(
        W_vanilla, data["W_true"], transgene_idx, "Vanilla",
        k=cfg["evaluation"]["top_k_transgene"], verbose=verbose)
    binn_tg = evaluate_transgene_targets(
        W_binn, data["W_true"], transgene_idx, "BINN",
        k=cfg["evaluation"]["top_k_transgene"], verbose=verbose)

    # ── 3. Identifiability ────────────────────────────────────────────
    id_results = None
    if not skip_stress_test:
        if verbose:
            print("\n" + "─" * 60)
            print("  [3/3]  Identifiability Stress Test")
            print("─" * 60)
        id_results = run_identifiability_stress_test(cfg, data, verbose=verbose)
    elif verbose:
        print("\n  [3/3]  Identifiability stress test SKIPPED "
              "(skip_stress_test=True)")

    # ── figures ────────────────────────────────────────────────────────
    if verbose:
        print("\n  Generating Phase 3 figures...")
    plot_extrapolation_comparison(vanilla_extra, binn_extra, data, labels,
                                  "figures/05_extrapolation_comparison.png")
    plot_roc_pr_curves(vanilla_rec, binn_rec, "figures/06_roc_pr_curves.png")
    if id_results is not None:
        plot_identifiability(id_results, cfg, "figures/07_identifiability_seeds.png")

    # ── save numeric results to CSV ───────────────────────────────────
    rows = [
        ["metric", "vanilla", "binn"],
        ["test_mse_noisy", vanilla_extra["mse_test"], binn_extra["mse_test"]],
        ["test_r2_noisy", vanilla_extra["r2_test"], binn_extra["r2_test"]],
        ["test_mse_clean", vanilla_extra.get("mse_test_clean", float("nan")),
         binn_extra.get("mse_test_clean", float("nan"))],
        ["test_r2_clean", vanilla_extra.get("r2_test_clean", float("nan")),
         binn_extra.get("r2_test_clean", float("nan"))],
        ["network_auroc", vanilla_rec["auroc"], binn_rec["auroc"]],
        ["network_aupr", vanilla_rec["aupr"], binn_rec["aupr"]],
        ["transgene_recall_at_k", vanilla_tg["recall_at_k"], binn_tg["recall_at_k"]],
    ]
    if id_results is not None:
        rows += [
            ["seed_auroc_mean", id_results["vanilla"]["aurocs"].mean(),
             id_results["binn"]["aurocs"].mean()],
            ["seed_auroc_std", id_results["vanilla"]["aurocs"].std(),
             id_results["binn"]["aurocs"].std()],
            ["seed_pairwise_corr", id_results["vanilla"]["mean_corr"],
             id_results["binn"]["mean_corr"]],
            ["transgene_jaccard", id_results["vanilla"]["jaccard"],
             id_results["binn"]["jaccard"]],
        ]

    with open("outputs/phase3_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # ── final printed summary ─────────────────────────────────────────
    if verbose:
        width = 58
        print("\n  ┌" + "─" * width + "┐")
        print("  │" + "  PHASE 3 SUMMARY".ljust(width) + "│")
        print("  ├" + "─" * width + "┤")
        for name, v, b in rows[1:]:
            try:
                print(f"  │  {name:26s} {float(v):>12.4f}  {float(b):>12.4f}  │")
            except (TypeError, ValueError):
                print(f"  │  {name:26s} {str(v):>12s}  {str(b):>12s}  │")
        print("  └" + "─" * width + "┘")
        print("\n  Saved: outputs/phase3_results.csv")
        print("  Saved: figures/05_extrapolation_comparison.png")
        print("  Saved: figures/06_roc_pr_curves.png")
        if id_results is not None:
            print("  Saved: figures/07_identifiability_seeds.png")
        print("\n" + "=" * 60)
        print("  ✓  Phase 3 complete")
        print("=" * 60)

    return {
        "vanilla_extra": vanilla_extra, "binn_extra": binn_extra,
        "vanilla_rec": vanilla_rec, "binn_rec": binn_rec,
        "vanilla_tg": vanilla_tg, "binn_tg": binn_tg,
        "id_results": id_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_phase3()