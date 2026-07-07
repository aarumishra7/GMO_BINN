"""
train.py — Phase 2: Train both models on the Phase 1 synthetic data.

Runnable standalone:
    python -m src.train

Trains the Vanilla Neural ODE and the BINN side-by-side on the same data,
then saves checkpoints, interaction matrices, and loss curves to outputs/.
"""

import time
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from src.utils import (load_config, seed_everything, ensure_dirs,
                        load_arrays, save_arrays, gene_labels)
from src.model_vanilla import VanillaODE
from src.model_binn import BINN_ODE


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_phase1_data(data_dir: str = "data") -> dict:
    """Load Phase 1 arrays and convert to PyTorch tensors."""
    arrays = load_arrays(
        data_dir,
        "x_train", "x_test", "t_train", "t_test",
        "grn_adjacency_true", "prior_adjacency",
    )
    return {
        "x_train": torch.tensor(arrays["x_train"], dtype=torch.float32),
        "x_test":  torch.tensor(arrays["x_test"],  dtype=torch.float32),
        "t_train": torch.tensor(arrays["t_train"], dtype=torch.float32),
        "t_test":  torch.tensor(arrays["t_test"],   dtype=torch.float32),
        "W_true":  arrays["grn_adjacency_true"],
        "prior":   arrays["prior_adjacency"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop  (one model at a time)
# ═══════════════════════════════════════════════════════════════════════════

def train_one_model(model, data: dict, cfg: dict,
                    model_name: str, is_binn: bool = False,
                    verbose: bool = True,
                    n_epochs_override: int | None = None) -> dict:
    """
    Train a single Neural ODE model.

    Parameters
    ----------
    verbose : if False, suppresses all per-epoch and summary printing.
        Used by the Phase 3 identifiability stress test, which trains
        many models back-to-back and would otherwise flood the terminal.
    n_epochs_override : if given, trains for this many epochs instead of
        cfg["training"]["n_epochs"]. Used by the stress test to keep the
        5-seeds-x-2-models retraining budget reasonable.

    Returns a history dict with per-epoch losses.
    """
    tcfg = cfg["training"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=tcfg["scheduler_patience"],
        factor=0.5, min_lr=1e-6,
    )

    # ── prepare tensors ──
    # y0 : initial condition  (n_conditions, n_genes)
    # t  : training times     (n_train_times,)
    # tgt: target trajectory  (n_train_times, n_conditions, n_genes)
    y0  = data["x_train"][:, 0, :]
    t   = data["t_train"]
    tgt = data["x_train"].permute(1, 0, 2)

    n_epochs   = n_epochs_override or tcfg["n_epochs"]
    best_loss  = float("inf")
    patience   = 0
    history    = {"total": [], "data": [], "lr": []}
    if is_binn:
        history["l1"] = []
        history["prior"] = []

    # ── header ──
    if verbose:
        cols = f"  {'Epoch':>6}  {'Total':>10}  {'DataMSE':>10}"
        if is_binn:
            cols += f"  {'L1':>9}  {'Prior':>9}"
        cols += f"  {'LR':>9}"
        print(f"\n  Training {model_name}  ({sum(p.numel() for p in model.parameters())} params)")
        print(cols)
        print("  " + "─" * (len(cols) - 2))

    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # ── forward: integrate ODE ──
        try:
            pred = odeint(model, y0, t, method="dopri5",
                          rtol=1e-5, atol=1e-6,
                          options={"max_num_steps": 2000})
            # torchdiffeq's stubs don't pin down odeint's return type, so
            # Pylance sees it as Unknown/tuple-shaped and flags the mse_loss
            # call below. It is always a single Tensor here (no event_fn
            # passed in), so this cast just tells the checker what we know.
            pred = cast(torch.Tensor, pred)
        except Exception as exc:
            if verbose and epoch % 100 == 0:
                print(f"  {epoch:6d}  solver error: {exc}")
            continue

        # ── losses ──
        data_loss  = F.mse_loss(pred, tgt)
        total_loss = data_loss.clone()

        l1_val = prior_val = 0.0
        if is_binn:
            l1    = tcfg["l1_lambda"]           * model.l1_penalty()
            prior = tcfg["prior_penalty_lambda"] * model.prior_penalty()
            total_loss = total_loss + l1 + prior
            l1_val, prior_val = l1.item(), prior.item()

        if not torch.isfinite(total_loss):
            if verbose and epoch % 100 == 0:
                print(f"  {epoch:6d}  non-finite loss, skipping step")
            continue

        # ── backward ──
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(total_loss.item())

        lr = optimizer.param_groups[0]["lr"]

        # ── bookkeeping ──
        history["total"].append(total_loss.item())
        history["data"].append(data_loss.item())
        history["lr"].append(lr)
        if is_binn:
            history["l1"].append(l1_val)
            history["prior"].append(prior_val)

        # ── logging ──
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            row = f"  {epoch:6d}  {total_loss.item():10.6f}  {data_loss.item():10.6f}"
            if is_binn:
                row += f"  {l1_val:9.5f}  {prior_val:9.5f}"
            row += f"  {lr:9.6f}"
            print(row)

        # ── early stopping ──
        if total_loss.item() < best_loss - 1e-7:
            best_loss = total_loss.item()
            patience = 0
        else:
            patience += 1
        if patience >= tcfg["early_stop_patience"]:
            if verbose:
                print(f"  → early stop at epoch {epoch}  "
                      f"(no improvement for {tcfg['early_stop_patience']} epochs)")
            break

    elapsed = time.time() - t0
    final_data = history["data"][-1] if history["data"] else float("nan")
    if verbose:
        print(f"  ✓ {model_name} done  |  best={best_loss:.6f}  "
              f"final_data_mse={final_data:.6f}  time={elapsed:.1f}s")

    return history


# ═══════════════════════════════════════════════════════════════════════════
# Interaction-matrix extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_interactions(model, data: dict, model_name: str,
                         is_binn: bool, verbose: bool = True) -> np.ndarray:
    """
    Get the (n_genes × n_genes) interaction matrix from a trained model.
    BINN reads W_reg directly; Vanilla uses averaged Jacobian.
    """
    model.eval()
    if is_binn:
        W = model.get_interaction_matrix()
    else:
        # flatten training data to (n_points, n_genes) for Jacobian
        x_flat = data["x_train"].reshape(-1, data["x_train"].shape[-1])
        W = model.get_interaction_matrix(reference_points=x_flat)
    if verbose:
        print(f"  {model_name} interaction matrix extracted  "
              f"(range [{W.min():.4f}, {W.max():.4f}])")
    return W


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_phase2(cfg: dict | None = None, verbose: bool = True) -> dict:
    """Train both models and save all artefacts."""
    if cfg is None:
        cfg = load_config()

    seed_everything(cfg["data"]["seed"])
    ensure_dirs("outputs")

    if verbose:
        print("=" * 60)
        print("  PHASE 2 — Model Training")
        print("=" * 60)

    # ── load data ──
    data = load_phase1_data()
    n_genes    = cfg["grn"]["n_total"]
    hidden_dim = cfg["model"]["hidden_dim"]
    n_layers   = cfg["model"]["n_hidden_layers"]

    if verbose:
        n_cond  = data["x_train"].shape[0]
        n_train = data["x_train"].shape[1]
        n_test  = data["x_test"].shape[1]
        print(f"\n  Data: {n_cond} conditions × {n_train} train + {n_test} test "
              f"timepoints × {n_genes} genes")

    # ─────────────────────────────────────────────────────────────────────
    # A.  Vanilla Neural ODE
    # ─────────────────────────────────────────────────────────────────────
    seed_everything(cfg["data"]["seed"])
    vanilla = VanillaODE(n_genes, hidden_dim, n_layers)
    vh = train_one_model(vanilla, data, cfg, "Vanilla Neural ODE", is_binn=False)

    # ─────────────────────────────────────────────────────────────────────
    # B.  BINN
    # ─────────────────────────────────────────────────────────────────────
    seed_everything(cfg["data"]["seed"])
    binn = BINN_ODE(n_genes, hidden_dim, data["prior"],
                    hill_n=cfg["dynamics"]["hill_n"],
                    hill_k=cfg["dynamics"]["hill_k"])
    bh = train_one_model(binn, data, cfg, "BINN (Biology-Informed)", is_binn=True)

    # ─────────────────────────────────────────────────────────────────────
    # Save everything
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print("\n  Saving artefacts to outputs/ ...")

    # model weights
    torch.save(vanilla.state_dict(), "outputs/vanilla_ode_model.pt")
    torch.save(binn.state_dict(),    "outputs/binn_model.pt")

    # interaction matrices
    W_vanilla = extract_interactions(vanilla, data, "Vanilla", is_binn=False)
    W_binn    = extract_interactions(binn,    data, "BINN",    is_binn=True)
    save_arrays("outputs",
                interaction_vanilla=W_vanilla,
                interaction_binn=W_binn)

    # loss histories
    np.savez("outputs/training_history.npz",
             vanilla_total=np.array(vh["total"]),
             vanilla_data=np.array(vh["data"]),
             vanilla_lr=np.array(vh["lr"]),
             binn_total=np.array(bh["total"]),
             binn_data=np.array(bh["data"]),
             binn_lr=np.array(bh["lr"]),
             binn_l1=np.array(bh.get("l1", [])),
             binn_prior=np.array(bh.get("prior", [])))

    if verbose:
        print("\n  Saved:")
        print("      outputs/vanilla_ode_model.pt")
        print("      outputs/binn_model.pt")
        print("      outputs/interaction_vanilla.npy")
        print("      outputs/interaction_binn.npy")
        print("      outputs/training_history.npz")
        print()

        # ── quick comparison ──
        print("  ┌────────────────────────────────────────────────┐")
        print("  │         Phase 2 Training Summary               │")
        print("  ├────────────────────────────────────────────────┤")
        vf = vh["data"][-1] if vh["data"] else float("nan")
        bf = bh["data"][-1] if bh["data"] else float("nan")
        print(f"  │  Vanilla final data MSE :  {vf:12.6f}         │")
        print(f"  │  BINN    final data MSE :  {bf:12.6f}         │")
        winner = "BINN" if bf < vf else "Vanilla"
        print(f"  │  Lower training loss    :  {winner:20s}     │")
        print("  └────────────────────────────────────────────────┘")
        print()
        print("=" * 60)
        print("  ✓  Phase 2 complete")
        print("=" * 60)

    return {
        "vanilla": vanilla, "binn": binn,
        "vh": vh, "bh": bh,
        "W_vanilla": W_vanilla, "W_binn": W_binn,
        "data": data,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_phase2()