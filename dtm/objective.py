"""Deletion regularizer and observed deletion rate (Sec. 3.2, 4.1).

    L_G = |V_t|^{-1} * sum_{j in V_t} g_j,        L = L_CE + alpha_t * L_G,
    r_t = |V_t|^{-1} * sum_{j in V_t} 1[g_j <= tau],

where ``V_t`` are the visual tokens of the batch, so the target ``delta`` is
the fraction of visual tokens removed. Since ``g_j < 0``, minimising ``L_G``
pushes scores toward ``c`` (deletion); ``alpha_t`` comes from the PID
controller (Eq. 3).
"""

from __future__ import annotations

import torch


def gate_loss(g: torch.Tensor, visual_mask: torch.Tensor) -> torch.Tensor:
    """``L_G``: mean gate score over the batch's visual tokens."""
    vis = visual_mask.to(torch.bool)
    if not bool(vis.any()):
        return g.new_zeros(())
    return g[vis].float().mean()


@torch.no_grad()
def observed_rate(g: torch.Tensor, visual_mask: torch.Tensor, tau: float) -> float:
    """``r_t``: fraction of the batch's visual tokens with ``g_j <= tau``."""
    vis = visual_mask.to(torch.bool)
    if not bool(vis.any()):
        return 0.0
    return float((g[vis] <= tau).float().mean())


def total_loss(ce_loss: torch.Tensor, g: torch.Tensor, visual_mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """``L = L_CE + alpha_t * L_G``."""
    return ce_loss + float(alpha) * gate_loss(g, visual_mask)


__all__ = ["gate_loss", "observed_rate", "total_loss"]
