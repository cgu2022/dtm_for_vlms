"""Attention with gate biases: soft deletion (Eq. 2) and answer-query biases (Sec. 4.3).

Eq. 2, for query i and key j in one head:

    s~_ij = q_i^T k_j / sqrt(d_h) + M_ij + g_j,
    A_ij  = exp(s~_ij) / (1 + sum_u exp(s~_iu))            (softmax_1)

* ``M_ij in {0, -inf}`` is the host model's attention mask (causal or
  bidirectional, local or global).
* ``g_j`` is the gate score of key ``j`` (0 for non-visual keys). During
  training the scores act as soft attention biases in every attention layer
  after the gate.
* softmax_1 adds 1 to the denominator so attention mass can dissipate when
  every key is suppressed; a standard softmax would be invariant to a uniform
  bias.

Threshold-shifted biases for answer queries (Sec. 4.3): encoder and prompt
queries use ``g_j``; decoder and answer queries use ``min(g_j - tau, 0)`` for
visual keys, and text keys receive no bias. Above-threshold visual keys thus
incur no penalty when the answer is generated, while below-threshold keys stay
suppressed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def softmax_1(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``exp(s_i) / (1 + sum_j exp(s_j))``, computed stably in fp32 [Eq. 2]."""
    s = scores.float()
    m = s.detach().amax(dim=dim, keepdim=True).clamp_min(torch.finfo(torch.float32).min)
    e = torch.exp(s - m)
    return e / (e.sum(dim=dim, keepdim=True) + torch.exp(-m))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, H_kv, S, d_h) -> (B, H_kv * n_rep, S, d_h) for grouped-query attention."""
    if n_rep == 1:
        return hidden_states
    b, h_kv, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


def eager_attention_forward_softmax_1(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention with softmax_1 (Eq. 2).

    Same signature and return value as the HuggingFace ``eager_attention_forward``
    of Gemma 3 / T5Gemma 2, so it can be registered as an attention backend.
    ``attention_mask`` is additive and already contains ``M_ij + g_j``.
    """
    if scaling is None:
        scaling = module.head_dim ** -0.5
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap
    if attention_mask is not None:
        mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + mask.clamp_min(torch.finfo(attn_weights.dtype).min).to(attn_weights.dtype)
    attn_weights = softmax_1(attn_weights, dim=-1).to(query.dtype)  # [Eq. 2]
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def key_bias(g: torch.Tensor) -> torch.Tensor:
    """Gate scores as an additive key-side bias, ``(B, N) -> (B, 1, 1, N)`` [Eq. 2]."""
    return g[:, None, None, :]


def answer_query_bias(g: torch.Tensor, visual_mask: torch.Tensor, tau: float) -> torch.Tensor:
    """Bias seen by decoder / answer queries [Sec. 4.3].

    ``min(g_j - tau, 0)`` for visual keys, ``0`` for text keys. Returns ``(B, N)``.
    """
    shifted = torch.clamp(g - tau, max=0.0)
    return torch.where(visual_mask.to(torch.bool), shifted, torch.zeros_like(shifted))


def add_key_bias(mask: Optional[torch.Tensor], bias: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Add a ``(B, N)`` key bias to an additive mask of shape ``(B, 1, Q, N)`` (or None)."""
    b = bias.to(dtype)[:, None, None, :]
    if mask is None:
        return b
    return mask.to(dtype) + b


__all__ = [
    "softmax_1",
    "repeat_kv",
    "eager_attention_forward_softmax_1",
    "key_bias",
    "answer_query_bias",
    "add_key_bias",
]
