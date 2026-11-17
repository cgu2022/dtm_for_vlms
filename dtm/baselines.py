"""Selection scores for the baselines (Sec. 5 "Baselines", App. D.6).

All methods delete visual tokens after five layers (L5, ``k = 6``) and keep
the ``n_v - floor(delta * n_v)`` highest-scoring visual tokens
(`dtm.deletion.exact_budget_delete_mask`). The ranking rules below are fixed;
no gradient flows through them.

* FastV: attention received by each visual key in layer ``k - 1``, reusing that
  layer's attention weights, averaged over heads and over a set of queries.
* phi-R: FEATHER's RoPE-free criterion. Scores are computed in layer ``k - 1``
  from that layer's projected, normalised queries and keys *before* the rotary
  transform, under the layer's attention mask, with the same aggregation. This
  counters the distance-dependent decay RoPE induces in FastV. FEATHER's
  uniform sampling and two-stage procedure are not used.
* Random: a uniformly random subset.

Query aggregation (App. D.6), heads averaged in every case:
  * ``all_mean``: all valid input queries (on Gemma 3, the prompt queries);
    used in the main results;
  * ``text_mean``: non-visual queries only;
  * ``text_last``: the final text query of the prompt.
"""

from __future__ import annotations

from typing import Optional

import torch

from .attention import repeat_kv


def query_rows(query_valid: torch.Tensor, text_query: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Bool ``(B, Q)``: which query rows are averaged for ``aggregation``."""
    valid = query_valid.to(torch.bool)
    if aggregation == "all_mean":
        return valid
    text = valid & text_query.to(torch.bool)
    if aggregation == "text_mean":
        return text
    if aggregation == "text_last":
        q = text.shape[1]
        pos = torch.arange(q, device=text.device).expand_as(text)
        last = torch.where(text, pos, torch.full_like(pos, -1)).amax(dim=1)  # index of last text query
        return (pos == last[:, None]) & text
    raise ValueError(f"unknown aggregation {aggregation!r}")


def aggregate_received_attention(
    attn: torch.Tensor, query_valid: torch.Tensor, text_query: torch.Tensor, aggregation: str
) -> torch.Tensor:
    """Mean attention received by each key over heads and the selected queries.

    Args:
        attn: ``(B, H, Q, K)`` attention probabilities.
        query_valid: ``(B, Q)`` bool, real (non-padding) query rows.
        text_query: ``(B, Q)`` bool, non-visual query rows.
        aggregation: ``"all_mean"``, ``"text_mean"`` or ``"text_last"``.

    Returns:
        ``(B, K)`` fp32 scores (higher = more attended = keep).
    """
    rows = query_rows(query_valid, text_query, aggregation)  # (B, Q)
    probs = attn.float()
    selected = torch.where(rows[:, None, :, None], probs, torch.zeros((), dtype=probs.dtype, device=probs.device))
    received = selected.sum(dim=(1, 2))  # (B, K)
    denom = (probs.shape[1] * rows.sum(dim=1, keepdim=True)).clamp(min=1).to(received.dtype)
    return received / denom


@torch.no_grad()
def fastv_scores(
    attn_weights: torch.Tensor, query_valid: torch.Tensor, text_query: torch.Tensor, aggregation: str = "all_mean"
) -> torch.Tensor:
    """FastV: received attention from layer ``k - 1``'s own attention weights."""
    return aggregate_received_attention(attn_weights, query_valid, text_query, aggregation)


@torch.no_grad()
def phi_r_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    query_valid: torch.Tensor,
    text_query: torch.Tensor,
    aggregation: str = "all_mean",
) -> torch.Tensor:
    """phi-R: RoPE-free received attention in layer ``k - 1``.

    Args:
        q: ``(B, H, Q, d_h)`` projected, normalised queries of layer ``k - 1``,
            before the rotary transform.
        k: ``(B, H_kv, K, d_h)`` projected, normalised keys, before the rotary
            transform.
        attention_mask: additive ``(B, 1, Q, K)`` mask of that layer, or None.
        scaling: the layer's score scaling.
    """
    k = repeat_kv(k, q.shape[1] // k.shape[1])
    scores = torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]].float().clamp_min(torch.finfo(torch.float32).min)
    probs = torch.softmax(scores, dim=-1)
    return aggregate_received_attention(probs, query_valid, text_query, aggregation)


def random_scores(visual_mask: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random: i.i.d. uniform scores, so the lowest ``floor(delta * n_v)`` form a uniform subset."""
    return torch.rand(visual_mask.shape, generator=generator, device=visual_mask.device)


__all__ = [
    "query_rows",
    "aggregate_received_attention",
    "fastv_scores",
    "phi_r_scores",
    "random_scores",
]
