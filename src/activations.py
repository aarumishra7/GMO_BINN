"""
activations.py — Shared activation functions for biology-informed models.
"""

import torch
import torch.nn as nn


class HillActivation(nn.Module):
    """
    Signed Hill-Langmuir activation.

        sigma(x) = sign(x) * |x|^n / (|x|^n + k^n)

    Saturates smoothly to +/-1. Biologically motivated: gene regulation
    has a bounded dose-response curve, not an unbounded linear one.
    """

    def __init__(self, n: float = 3.0, k: float = 0.5) -> None:
        super().__init__()
        self.n = n
        self.kn = k ** n  # precompute k^n once

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = x.abs() + 1e-8  # numerical safety, avoids 0^n grad issues
        xn = abs_x.pow(self.n)
        return x.sign() * xn / (xn + self.kn)