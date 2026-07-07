"""
model_vanilla.py — Model A: Vanilla Neural ODE (black-box baseline).

A fully connected MLP that outputs dg/dt, with no biological priors,
no structural constraints, and no interpretable internal weights.
Used as the "no domain knowledge" comparison point against the BINN.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class VanillaODE(nn.Module):
    """
    Fully connected MLP that outputs dg/dt.
    No biological priors, no structural constraints.
    """

    def __init__(self, n_genes: int, hidden_dim: int, n_layers: int = 2) -> None:
        super().__init__()
        self.n_genes = n_genes

        layers: list[nn.Module] = []
        in_dim = n_genes
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())          # smooth, bounded — helps ODE stability
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_genes))
        self.net = nn.Sequential(*layers)

        # small init for stable ODE integration at the start of training
        self._init_weights()

    # ── helpers ──────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """dg/dt.  g shape: (batch, n_genes) or (n_genes,)."""
        return self.net(g)

    # ── interaction extraction ───────────────────────────────────────────

    @torch.no_grad()
    def get_interaction_matrix(
        self, reference_points: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Average absolute Jacobian |df_i/dg_j| over reference points.

        For a black box this is the only principled way to read out which
        input genes influence which output derivatives — there is no
        directly readable weight matrix like the BINN's W_reg.
        """
        if reference_points is None:
            raise ValueError("VanillaODE needs reference_points for Jacobian")

        self.eval()
        n = self.n_genes
        n_pts = min(64, len(reference_points))
        J_sum = torch.zeros(n, n)

        for idx in range(n_pts):
            g = reference_points[idx].clone().detach().requires_grad_(True)
            with torch.enable_grad():
                f = self.forward(torch.tensor(0.0), g)
                for i in range(n):
                    grad_i = torch.autograd.grad(
                        f[i], g, retain_graph=(i < n - 1)
                    )[0]
                    J_sum[i] += grad_i.abs()

        return (J_sum / n_pts).numpy()