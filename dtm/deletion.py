"""Hard deletion: which visual tokens to remove, and physical compaction.

* Natural deletion (Sec. 3.2): at inference, states with ``g_j <= tau`` are
  physically removed, so all later layers process a shorter sequence.
* Exact budget (Sec. 5 "Two evaluation settings"; App. C "Inference budget"):
  every method ranks the ``n_v`` visual tokens of an example by its score and
  deletes exactly ``floor(delta * n_v)`` of them, the lowest-scoring first.
  Ties are broken by position (a stable ranking), so the count is always exact.
* Retained tokens keep their original position indices (Sec. 4.1, 4.2), so
  rotary embeddings and local-window / causal visibility are evaluated in the
  original position space.
"""

from __future__ import annotations

import torch


def natural_delete_mask(g: torch.Tensor, visual_mask: torch.Tensor, tau: float) -> torch.Tensor:
    """True where a visual token has ``g_j <= tau`` [Sec. 3.2]. Shape ``(B, N)``."""
    return visual_mask.to(torch.bool) & (g <= tau)


def budget(n_visual: torch.Tensor, delta: float) -> torch.Tensor:
    """``floor(delta * n_v)`` per example."""
    delta = float(min(max(delta, 0.0), 1.0))
    k = torch.floor(n_visual.to(torch.float32) * delta).to(torch.long)
    return torch.minimum(k, n_visual.to(torch.long))


def exact_budget_delete_mask(scores: torch.Tensor, visual_mask: torch.Tensor, delta: float) -> torch.Tensor:
    """Delete exactly ``floor(delta * n_v)`` lowest-scoring visual tokens per row.

    Args:
        scores: ``(B, N)`` selection scores (higher = keep). DTM uses its gate
            scores ``g``; FastV / phi-R use received attention; Random uses
            uniform noise.
        visual_mask: ``(B, N)`` bool deletion candidates.
        delta: deletion rate.

    Returns:
        ``(B, N)`` bool, True = delete.
    """
    cand = visual_mask.to(torch.bool)
    k = budget(cand.sum(dim=1), delta)
    masked = scores.float().masked_fill(~cand, float("inf"))  # non-candidates rank last
    ranks = masked.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
    return cand & (ranks < k.unsqueeze(1))


def compact(x: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Front-pack the kept positions of each row, preserving their order.

    Args:
        x: ``(B, N, ...)``.
        keep: ``(B, N)`` bool.

    Returns:
        ``(packed, index, valid)``: ``packed`` is ``(B, N', ...)`` with
        ``N' = max_b sum_j keep[b, j]``; ``index`` is ``(B, N')`` long, the
        original position of every packed slot; ``valid`` is ``(B, N')`` bool,
        False on the right-padding of shorter rows.
    """
    keep = keep.to(torch.bool)
    counts = keep.sum(dim=1)
    n_out = int(counts.max().item()) if keep.numel() else 0
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)  # kept first, original order
    index = order[:, :n_out]
    valid = torch.arange(n_out, device=keep.device)[None, :] < counts[:, None]
    return gather_seq(x, index), index, valid


def gather_seq(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x[b, index[b, j], ...]`` for ``x`` of shape ``(B, N, ...)``."""
    idx = index.reshape(*index.shape, *([1] * (x.dim() - 2))).expand(*index.shape, *x.shape[2:])
    return torch.gather(x, 1, idx)


__all__ = ["natural_delete_mask", "budget", "exact_budget_delete_mask", "compact", "gather_seq"]
