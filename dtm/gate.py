"""The DTM delete gate (Eq. 1) and its initial deletion-rate calibration (App. A.2).

Eq. 1:   g = c * sigmoid( -( RMSNorm(H) w + b ) ),   g in (c, 0)^n,   c < 0.

* ``H`` holds the hidden states entering layer ``k`` (the output of layer
  ``k - 1``); ``w in R^d`` and ``b in R`` are learned; RMSNorm has its own
  learned scale. The gate has ``2d + 1`` parameters (``d`` RMSNorm scale,
  ``d`` projection weights, one bias).
* A score near 0 keeps a token; a score near ``c`` suppresses it. A token is
  deleted when ``g_j <= tau``.
* Vision-only deletion (Sec. 4.1): the gate scores visual tokens only; text
  tokens and image-boundary markers receive ``g_j = 0`` and are never deleted.

Calibration (Sec. 4.3, App. A.2, Eq. 6): once, on the first training
batch, the bias is set to

    b0 = -logit(tau / c) - q0,

where ``q0`` is the empirical ``delta_init``-quantile of the projection scores
``z_j = w^T x_j`` of that batch's visual tokens (``x_j`` the normalised hidden
state). Then ``g_j <= tau`` exactly when ``z_j <= q0``, so a fraction
``delta_init`` of the calibration tokens starts below the threshold.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GateRMSNorm(nn.Module):
    """RMSNorm with a (1 + weight) scale, as in the Gemma 3 / T5Gemma 2 blocks."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # scale = 1 + weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x32 * (1.0 + self.weight.float())


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class DeleteGate(nn.Module):
    """Scalar delete gate over visual hidden states (Eq. 1).

    Args:
        hidden_size: model width ``d``.
        c: gate scale (``c = -22`` in the paper).
        tau: deletion threshold (``tau = -7`` in the paper).
        eps: RMSNorm epsilon.

    Parameters are kept in fp32; inputs of any dtype are accepted.
    """

    def __init__(self, hidden_size: int, c: float = -22.0, tau: float = -7.0, eps: float = 1e-6) -> None:
        super().__init__()
        if not (c < tau < 0.0):
            raise ValueError(f"need c < tau < 0, got c={c}, tau={tau}")
        self.c = float(c)
        self.tau = float(tau)
        self.norm = GateRMSNorm(hidden_size, eps=eps)  # RMSNorm with its own scale [Eq. 1]
        self.proj = nn.Linear(hidden_size, 1)  # w and b [Eq. 1]
        # Initial weights: w ~ N(0, 0.02^2), b = 0; RMSNorm scale 1 (weight 0).
        # The bias is then set by `calibrate_` on the first training batch.
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.register_buffer("calibrated", torch.tensor(False), persistent=True)
        self.float()

    # ------------------------------------------------------------------ #
    def projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``z = w^T RMSNorm(h)`` (no bias), in fp32."""
        x = self.norm(hidden_states)
        return torch.nn.functional.linear(x, self.proj.weight.float()).squeeze(-1)

    def forward(self, hidden_states: torch.Tensor, visual_mask: torch.Tensor) -> torch.Tensor:
        """Gate scores ``g`` for every position.

        Args:
            hidden_states: ``(B, N, d)`` states entering layer ``k``.
            visual_mask: ``(B, N)`` bool, True at visual tokens (the deletion
                candidates). Text tokens, image-boundary markers and padding
                are False.

        Returns:
            ``(B, N)`` fp32 scores; ``g in (c, 0)`` at visual tokens and
            ``g = 0`` everywhere else [Sec. 4.1].
        """
        visual_mask = visual_mask.to(torch.bool)
        g = hidden_states.new_zeros(visual_mask.shape, dtype=torch.float32)
        if bool(visual_mask.any()):
            x = hidden_states[visual_mask]  # (n_visual, d): gate only visual tokens
            z = self.projection(x) + self.proj.bias.float()  # RMSNorm(H) w + b
            g_visual = self.c * torch.sigmoid(-z)  # [Eq. 1]
            g = g.masked_scatter(visual_mask, g_visual)
        return g

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def calibrate_(self, hidden_states: torch.Tensor, visual_mask: torch.Tensor, delta_init: float) -> float:
        """One-time bias calibration on the first training batch [App. A.2].

        Sets ``b0 = -logit(tau/c) - q0`` with ``q0`` the ``delta_init``-quantile
        of ``z_j = w^T x_j`` over the batch's visual tokens [Eq. 6],
        holding the initial ``w`` and RMSNorm scale fixed. Returns ``b0``.
        """
        if not (0.0 <= delta_init <= 1.0):
            raise ValueError(f"delta_init must be in [0, 1], got {delta_init}")
        z = self.projection(hidden_states[visual_mask.to(torch.bool)])
        if z.numel() == 0:
            raise ValueError("calibration batch contains no visual tokens")
        q0 = float(torch.quantile(z.float(), float(delta_init)))
        b0 = -logit(self.tau / self.c) - q0  # [Eq. 6]
        self.proj.bias.fill_(b0)
        self.calibrated.fill_(True)
        return b0

    def extra_repr(self) -> str:
        return f"c={self.c}, tau={self.tau}"


__all__ = ["DeleteGate", "GateRMSNorm", "logit"]
