"""
utils.py — Shared helpers: config loading, seeding, I/O, plotting palette.
"""

import os
import random
import yaml
import numpy as np
import torch


# ── Config ────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config, resolve derived fields."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["grn"]["n_total"] = cfg["grn"]["n_native_genes"] + cfg["grn"]["n_transgenes"]
    return cfg


# ── Reproducibility ──────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Directory helpers ────────────────────────────────────────────────────

def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ── Save / Load ──────────────────────────────────────────────────────────

def save_arrays(directory: str, **named_arrays):
    """Save NumPy arrays to *directory* with filenames from keyword names."""
    ensure_dirs(directory)
    paths = {}
    for name, arr in named_arrays.items():
        p = os.path.join(directory, f"{name}.npy")
        np.save(p, arr)
        paths[name] = p
    return paths


def load_arrays(directory: str, *names):
    """Load previously saved arrays by name."""
    return {n: np.load(os.path.join(directory, f"{n}.npy")) for n in names}


# ── Gene / label helpers ─────────────────────────────────────────────────

def gene_labels(n_native: int, n_transgene: int = 1) -> list[str]:
    """Return human-readable gene labels."""
    native = [f"Gene_{i}" for i in range(n_native)]
    trans  = [f"Transgene_{j}" for j in range(n_transgene)]
    return native + trans