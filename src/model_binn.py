"""
model_binn.py — Model B: Biology-Informed Neural ODE (BINN).

Three biology-informed modifications distinguish this from the vanilla
Neural ODE in model_vanilla.py:

  1. **Interpretable regulatory layer**  W_reg in R^(n x n)
     W_reg[i, j] = influence of gene j on gene i.
     Directly readable as a candidate interaction graph — no Jacobian
     extraction needed, unlike the vanilla model.

  2. **Hill-Langmuir activations** instead of ReLU/Tanh —
     encodes the biological prior that regulation saturates.

  3. **Biophysical decomposition**:
        dg/dt = basal + transcription(hill(W_reg . g)) - delta * g
     Positive transcription branch is shaped by the prior network;
     per-gene degradation rate delta is learned independently.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.activations import HillActivation


class BINN_ODE(nn.Module):
    """
    Biology-Informed Neural ODE.

    Penalty API:
      * l1_penalty()    — global L1 on W_reg  (sparsity)
      * prior_penalty() — extra L1 on edges absent from the prior database
    """

    # Explicit buffer type annotations. Without these, static type checkers
    # (Pylance) infer attributes registered via register_buffer as a
    # `Tensor | Module` union (matching nn.Module.__getattr__'s general
    # return type), which then makes `self.W_reg * self.diag_mask` look like
    # an unsupported operator between Parameter and Module. Declaring the
    # attribute type up front tells the checker these are always Tensor.
    diag_mask: torch.Tensor
    anti_prior: torch.Tensor

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        prior_adj: np.ndarray,
        hill_n: float = 3.0,
        hill_k: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        # ── constant masks (not trainable) ──
        diag = torch.eye(n_genes)
        prior_t = torch.tensor(prior_adj, dtype=torch.float32)
        self.register_buffer("diag_mask", 1.0 - diag)
        self.register_buffer("anti_prior",
                             (1.0 - prior_t) * (1.0 - diag))  # no penalty on diagonal

        # ── interpretable regulatory weights ──
        self.W_reg = nn.Parameter(torch.randn(n_genes, n_genes) * 0.05)
        with torch.no_grad():
            self.W_reg.fill_diagonal_(0.0)

        # ── Hill activation ──
        self.hill = HillActivation(n=hill_n, k=hill_k)

        # ── transcription sub-network  (positive output) ──
        self.trans_net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            HillActivation(n=hill_n, k=hill_k),
            nn.Linear(hidden_dim, n_genes),
            nn.Softplus(),                     # guarantees positive
        )

        # ── per-gene basal expression  (positive) ──
        self.log_basal = nn.Parameter(torch.randn(n_genes) * 0.1 - 1.0)

        # ── per-gene degradation rate  (positive) ──
        self.log_delta = nn.Parameter(torch.randn(n_genes) * 0.1 - 0.5)

        self._init_weights()

    # ── helpers ──────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.trans_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        dg/dt = basal + transcription(hill(W_reg . g)) - delta * g
        """
        # mask out self-regulation every forward pass
        W = self.W_reg * self.diag_mask

        # regulatory signal per gene: g @ W^T -> (batch, n_genes)
        reg = self.hill(F.linear(g, W))

        # positive transcription rate
        trans = self.trans_net(reg)

        # biophysical assembly
        basal = F.softplus(self.log_basal)
        degrad = F.softplus(self.log_delta) * g

        return basal + trans - degrad

    # ── penalty API ──────────────────────────────────────────────────────

    def l1_penalty(self) -> torch.Tensor:
        """Global L1 on off-diagonal regulatory weights."""
        return (self.W_reg.abs() * self.diag_mask).sum()

    def prior_penalty(self) -> torch.Tensor:
        """Extra L1 on edges *not* present in the prior database."""
        return (self.W_reg.abs() * self.anti_prior).sum()

    # ── interaction extraction ───────────────────────────────────────────

    @torch.no_grad()
    def get_interaction_matrix(
        self, reference_points: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Directly return W_reg — the whole point of a BINN is that
        the interaction weights are interpretable without Jacobian tricks.
        `reference_points` is accepted (and ignored) so both models share
        the same call signature in train.py.
        """
        W = self.W_reg.cpu().numpy().copy()
        np.fill_diagonal(W, 0.0)
        return W