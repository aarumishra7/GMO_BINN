"""
grn_simulator.py — Phase 1: Build a ground-truth GRN and generate synthetic
time-course expression data for multiple GMO insertion contexts.

Runnable standalone:
    python -m src.grn_simulator
"""

import numpy as np
from scipy.integrate import solve_ivp
from src.utils import load_config, seed_everything, save_arrays, ensure_dirs, gene_labels


# ═══════════════════════════════════════════════════════════════════════════
# 1. GROUND-TRUTH NETWORK
# ═══════════════════════════════════════════════════════════════════════════

def build_adjacency(n_native: int,
                    n_transgene: int,
                    edge_density: float,
                    activation_ratio: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Create a sparse, signed adjacency matrix W  (n_total × n_total).
    
    W[i, j] != 0 means gene j regulates gene i.
    Positive = activation, negative = repression.
    The transgene column (last) always has at least 2 outgoing edges
    into native genes so its effect is detectable.
    """
    n = n_native + n_transgene

    # --- random sparse edges among native genes ---
    mask = rng.random((n, n)) < edge_density
    np.fill_diagonal(mask, False)           # no self-regulation
    mask[:, n_native:] = False              # clear transgene column; set manually below

    # --- transgene → native edges (guarantee ≥2) ---
    n_trans_targets = max(2, int(n_native * edge_density))
    targets = rng.choice(n_native, size=n_trans_targets, replace=False)
    for t in targets:
        mask[t, n_native] = True

    # --- no native → transgene edges (constitutive promoter) ---
    mask[n_native:, :n_native] = False

    # --- assign signs ---
    W = np.zeros((n, n), dtype=np.float64)
    edge_rows, edge_cols = np.where(mask)
    for r, c in zip(edge_rows, edge_cols):
        sign = 1.0 if rng.random() < activation_ratio else -1.0
        magnitude = rng.uniform(0.2, 1.0)
        W[r, c] = sign * magnitude

    return W


# ═══════════════════════════════════════════════════════════════════════════
# 2. ODE DYNAMICS  (Hill-kinetics)
# ═══════════════════════════════════════════════════════════════════════════

def hill(x: np.ndarray, n: float, k: float) -> np.ndarray:
    """Positive Hill function: x^n / (x^n + k^n)."""
    xn = np.power(np.clip(x, 0, None), n)
    return xn / (xn + k**n + 1e-12)


def make_ode_fn(W, basal, degradation, hill_n, hill_k, transgene_basal, n_native):
    """
    Return the RHS  dg/dt = f(t, g)  for the GRN ODE.

    dg_i/dt = basal_i
              + Σ_j [ W_ij * hill(g_j)  if W_ij > 0 ]     (activation)
              + Σ_j [ W_ij * (1 - hill(g_j))  if W_ij < 0 ] (repression)
              - δ_i * g_i                                    (degradation)

    The transgene has its own strong constitutive basal rate.
    """
    def f(t, g):
        g = np.clip(g, 0, None)           # concentrations can't go negative
        h = hill(g, hill_n, hill_k)
        regulation = np.zeros_like(g)
        for i in range(len(g)):
            for j in range(len(g)):
                w = W[i, j]
                if w > 0:
                    regulation[i] += w * h[j]
                elif w < 0:
                    regulation[i] += w * (1.0 - h[j])
        dgdt = basal + regulation - degradation * g
        # override transgene basal (strong constitutive promoter)
        dgdt[n_native:] = transgene_basal - degradation[n_native:] * g[n_native:]
        # transgene still regulates natives via W; its own level is set by its promoter
        return dgdt
    return f


# ═══════════════════════════════════════════════════════════════════════════
# 3. DATA GENERATION  (multiple GMO insertion contexts)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_condition(W, basal, degradation, hill_n, hill_k,
                       transgene_basal, n_native, t_span, n_timepoints,
                       rng):
    """
    Integrate the ODE for one GMO line / condition.
    Returns (t_eval, expression matrix  [n_timepoints × n_genes]).
    """
    n = W.shape[0]
    g0 = rng.uniform(0.0, 0.3, size=n)           # low random initial expression
    g0[n_native:] = transgene_basal * 0.5         # transgene starts at ~half steady state

    t_eval = np.linspace(t_span[0], t_span[1], n_timepoints)
    ode_fn = make_ode_fn(W, basal, degradation, hill_n, hill_k,
                         transgene_basal, n_native)

    sol = solve_ivp(ode_fn, t_span, g0, t_eval=t_eval,
                    method="RK45", max_step=0.05, rtol=1e-8, atol=1e-10)

    if sol.status != 0:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    expr = np.clip(sol.y.T, 0, None)              # enforce non-negative concentrations
    return t_eval, expr                            # shape (n_timepoints, n_genes)


def generate_conditions(W, cfg, rng):
    """
    Generate expression data for *n_conditions* GMO lines.
    Each condition perturbs the transgene copy-number (scales its basal rate)
    and slightly jitters degradation rates (insertion-site effects).

    Returns
    -------
    t_eval        : (n_timepoints,)
    clean_data    : (n_conditions, n_timepoints, n_genes)  — noise-free
    """
    dyn  = cfg["dynamics"]
    data = cfg["data"]
    n_native = cfg["grn"]["n_native_genes"]
    n_total  = cfg["grn"]["n_total"]

    basal = rng.uniform(*dyn["basal_expression_range"], size=n_total)
    degradation = rng.uniform(*dyn["degradation_range"], size=n_total)

    copy_number_scales = np.linspace(0.5, 2.0, data["n_conditions"])  # 0.5× to 2× copy number

    # t_eval is identical across every condition, so compute it once up front.
    # (Previously this was assigned inside the loop from simulate_condition's
    # return value, starting from None — Pylance couldn't prove the loop runs
    # at least once, so every downstream len(t_eval) was flagged as possibly
    # operating on None. Computing it directly removes the Optional entirely.)
    t_eval = np.linspace(dyn["t_span"][0], dyn["t_span"][1], dyn["n_timepoints"])

    all_expr = []

    for ci, cn_scale in enumerate(copy_number_scales):
        # per-condition transgene basal = base × copy-number scale
        tb = dyn["transgene_basal"] * cn_scale

        # slight per-condition jitter on degradation (insertion-site effect)
        deg_jitter = degradation * (1.0 + rng.normal(0, 0.05, size=n_total))
        deg_jitter = np.clip(deg_jitter, 0.01, None)

        _, expr = simulate_condition(
            W, basal, deg_jitter,
            dyn["hill_n"], dyn["hill_k"], tb, n_native,
            dyn["t_span"], dyn["n_timepoints"], rng,
        )
        all_expr.append(expr)

    clean_data = np.stack(all_expr, axis=0)       # (C, T, G)
    return t_eval, clean_data, basal, degradation


# ═══════════════════════════════════════════════════════════════════════════
# 4. NOISE + DROPOUT
# ═══════════════════════════════════════════════════════════════════════════

def add_noise(clean: np.ndarray, noise_std: float, dropout_rate: float,
              rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian noise and random dropout (set entries to 0) to mimic
    real single-cell-like measurement artefacts.
    """
    noisy = clean + rng.normal(0, noise_std, size=clean.shape)
    noisy = np.clip(noisy, 0, None)               # expression ≥ 0

    # dropout mask
    drop_mask = rng.random(clean.shape) < dropout_rate
    noisy[drop_mask] = 0.0

    return noisy


# ═══════════════════════════════════════════════════════════════════════════
# 5. IMPERFECT PRIOR NETWORK
# ═══════════════════════════════════════════════════════════════════════════

def build_prior(W_true: np.ndarray,
                edge_drop_rate: float,
                false_edge_rate: float,
                rng: np.random.Generator) -> np.ndarray:
    """
    Build an imperfect binary prior adjacency from the true W.
    - Randomly remove *edge_drop_rate* fraction of true edges.
    - Randomly add *false_edge_rate* fraction of spurious edges.
    Returns a binary matrix (1 = edge present in prior).
    """
    true_binary = (W_true != 0).astype(np.float64)
    prior = true_binary.copy()

    # drop real edges
    real_edges = list(zip(*np.where(true_binary == 1)))
    n_drop = int(len(real_edges) * edge_drop_rate)
    drop_idx = rng.choice(len(real_edges), size=n_drop, replace=False)
    for di in drop_idx:
        r, c = real_edges[di]
        prior[r, c] = 0.0

    # add false edges
    zero_entries = list(zip(*np.where(true_binary == 0)))
    # exclude diagonal
    zero_entries = [(r, c) for r, c in zero_entries if r != c]
    n_false = int(len(zero_entries) * false_edge_rate)
    false_idx = rng.choice(len(zero_entries), size=n_false, replace=False)
    for fi in false_idx:
        r, c = zero_entries[fi]
        prior[r, c] = 1.0

    return prior


# ═══════════════════════════════════════════════════════════════════════════
# 6. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def split_time(t_eval, data, train_frac):
    """
    Split along time axis. First train_frac of timepoints → train,
    remainder → test (for extrapolation evaluation).
    """
    n_train = int(len(t_eval) * train_frac)
    return {
        "t_train": t_eval[:n_train],
        "t_test":  t_eval[n_train:],
        "x_train": data[:, :n_train, :],
        "x_test":  data[:, n_train:, :],
        "n_train": n_train,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN  — run Phase 1 end-to-end
# ═══════════════════════════════════════════════════════════════════════════

def run_phase1(cfg: dict | None = None, verbose: bool = True):
    """
    Execute the full Phase 1 pipeline and save everything to data/.

    Returns a dict with all generated artefacts for downstream use.
    """
    if cfg is None:
        cfg = load_config()

    seed_everything(cfg["data"]["seed"])
    rng = np.random.default_rng(cfg["data"]["seed"])

    data_dir = "data"
    ensure_dirs(data_dir)

    if verbose:
        print("=" * 60)
        print("  PHASE 1 — Ground-Truth GRN & Synthetic Expression Data")
        print("=" * 60)

    # ── 1. Adjacency matrix ──
    n_nat = cfg["grn"]["n_native_genes"]
    n_tg  = cfg["grn"]["n_transgenes"]
    n_tot = cfg["grn"]["n_total"]

    W = build_adjacency(n_nat, n_tg,
                        cfg["grn"]["edge_density"],
                        cfg["grn"]["activation_ratio"], rng)

    n_edges = int(np.count_nonzero(W))
    n_act   = int(np.sum(W > 0))
    n_rep   = int(np.sum(W < 0))
    if verbose:
        print(f"\n[1/6] Ground-truth adjacency built")
        print(f"      {n_tot} genes  ({n_nat} native + {n_tg} transgene)")
        print(f"      {n_edges} edges  ({n_act} activating, {n_rep} repressing)")

    # ── 2. Simulate expression ──
    t_eval, clean_data, basal, degradation = generate_conditions(W, cfg, rng)
    if verbose:
        n_cond = cfg["data"]["n_conditions"]
        print(f"\n[2/6] ODE simulation complete")
        print(f"      {n_cond} GMO conditions  (copy-number scale 0.5×–2.0×)")
        print(f"      {len(t_eval)} timepoints over [{cfg['dynamics']['t_span'][0]}, "
              f"{cfg['dynamics']['t_span'][1]}]")
        print(f"      Expression range: [{clean_data.min():.4f}, {clean_data.max():.4f}]")

    # ── 3. Add noise ──
    noisy_data = add_noise(clean_data,
                           cfg["data"]["noise_std"],
                           cfg["data"]["dropout_rate"], rng)
    if verbose:
        pct_zero = 100.0 * np.mean(noisy_data == 0)
        print(f"\n[3/6] Noise added")
        print(f"      Gaussian σ = {cfg['data']['noise_std']}")
        print(f"      Dropout rate = {cfg['data']['dropout_rate']}")
        print(f"      Zero entries after noise+dropout: {pct_zero:.1f}%")

    # ── 4. Imperfect prior ──
    prior = build_prior(W, cfg["prior"]["edge_drop_rate"],
                        cfg["prior"]["false_edge_rate"], rng)

    true_edges = set(zip(*np.where(W != 0)))
    prior_edges = set(zip(*np.where(prior != 0)))
    kept    = len(true_edges & prior_edges)
    dropped = len(true_edges - prior_edges)
    spurious = len(prior_edges - true_edges)
    if verbose:
        print(f"\n[4/6] Imperfect prior network built")
        print(f"      True edges kept in prior:   {kept}/{n_edges}")
        print(f"      True edges dropped:         {dropped}")
        print(f"      Spurious edges added:       {spurious}")

    # ── 5. Train/test split ──
    split = split_time(t_eval, noisy_data, cfg["data"]["train_time_fraction"])
    if verbose:
        print(f"\n[5/6] Train/test split")
        print(f"      Train timepoints: {len(split['t_train'])}  "
              f"(t ∈ [{split['t_train'][0]:.1f}, {split['t_train'][-1]:.1f}])")
        print(f"      Test  timepoints: {len(split['t_test'])}  "
              f"(t ∈ [{split['t_test'][0]:.1f}, {split['t_test'][-1]:.1f}])  ← extrapolation")

    # ── 6. Save ──
    paths = save_arrays(
        data_dir,
        grn_adjacency_true=W,
        prior_adjacency=prior,
        expression_clean=clean_data,
        expression_noisy=noisy_data,
        time_points=t_eval,
        t_train=split["t_train"],
        t_test=split["t_test"],
        x_train=split["x_train"],
        x_test=split["x_test"],
        basal_rates=basal,
        degradation_rates=degradation,
    )

    if verbose:
        print(f"\n[6/6] All arrays saved to {data_dir}/")
        for name, p in sorted(paths.items()):
            arr = np.load(p)
            print(f"      {name:25s}  shape={str(arr.shape):22s}  dtype={arr.dtype}")

    if verbose:
        print("\n" + "=" * 60)
        print("  ✓  Phase 1 complete")
        print("=" * 60)

    return {
        "W": W, "prior": prior,
        "clean_data": clean_data, "noisy_data": noisy_data,
        "t_eval": t_eval, "split": split,
        "basal": basal, "degradation": degradation,
        "labels": gene_labels(n_nat, n_tg),
        "cfg": cfg,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_phase1()