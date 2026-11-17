"""Analytical forward-pass FLOPs (App. G).

Conventions (App. G, "Notation and counting convention"):
* two FLOPs per multiply-add; dominant matrix multiplications of the language
  backbone and the vocabulary projection only (SigLIP and the multimodal
  projector are excluded; norms, RoPE, softmax, activations, residuals,
  ranking and gathering are omitted);
* grouped-query attention: Q/O projections scale with ``n_heads``, K/V with
  ``n_kv``; both attention products (QK^T, AV) scale with ``n_heads``;
* local layers count ``N * min(N, W)`` query-key pairs; global layers ``N^2``;
  the decoder's self-attention block uses the full-square ``M^2`` proxy;
* T5Gemma 2 decoder layers use merged attention: every decoder query attends
  to all encoder keys, even in local layers, and each decoder layer projects
  the encoder states into its own K/V.

Pruning (App. G.4): with deletion after layer
``k`` (``k = 5`` for L5), layers ``1..k`` process the full input and layers
``k+1..L`` (and, for T5Gemma 2, the decoder) the retained length. Only visual
tokens are removed. Selection costs: DTM adds ``2 N_v d`` (one scalar
projection per visual token); FastV reuses the attention of layer ``k`` and
adds nothing; phi-R adds one RoPE-free score contraction
``2 P(N, k) n_heads d_head``.

Savings (Table 2, gray row, App. F.1 tables): ``1 - sum F_method / sum F_base``
over the evaluated examples, at their recorded input and generated lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class StackConfig:
    layers: int
    d_model: int
    d_ff: int
    n_heads: int
    n_kv: int
    d_head: int
    window: int
    global_every: int = 6  # layers 6, 12, 18, ... are global (1-indexed)

    def is_global(self, i: int) -> bool:
        """``i`` is the 1-indexed layer number."""
        return i % self.global_every == 0


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    kind: str  # "t5gemma2" (encoder-decoder) or "gemma3" (decoder-only)
    encoder: StackConfig  # for Gemma 3: the single stack
    decoder: StackConfig | None
    vocab: int


# Table 7 (App. B.1).
T5GEMMA2_1B = BackboneConfig(
    "t5gemma2-1b-1b", "t5gemma2",
    StackConfig(26, 1152, 6912, 4, 1, 256, 512), StackConfig(26, 1152, 6912, 4, 1, 256, 512), 262_144)
T5GEMMA2_270M = BackboneConfig(
    "t5gemma2-270m-270m", "t5gemma2",
    StackConfig(18, 640, 2048, 4, 1, 256, 512), StackConfig(18, 640, 2048, 4, 1, 256, 512), 262_144)
GEMMA3_4B = BackboneConfig(
    "gemma3-4b", "gemma3", StackConfig(34, 2560, 10240, 8, 4, 256, 1024), None, 262_208)
BACKBONES = {b.name: b for b in (T5GEMMA2_1B, T5GEMMA2_270M, GEMMA3_4B)}


# --------------------------------------------------------------------------- #
# Per-layer costs
# --------------------------------------------------------------------------- #
def self_pairs(s: StackConfig, n: float, i: int) -> float:
    """Query-key pairs per head: N^2 (global) or N * min(N, W) (local) [Eq. 10]."""
    return n * n if s.is_global(i) else n * min(n, s.window)


def layer_flops(s: StackConfig, n: float, i: int) -> float:
    """Encoder layer / Gemma 3 layer [Eq. 12, flops_gemma_layer]."""
    proj = 4 * n * s.d_model * s.d_head * (s.n_heads + s.n_kv)  # Q, K, V, O projections
    attn = 4 * self_pairs(s, n, i) * s.n_heads * s.d_head  # QK^T and AV
    mlp = 6 * n * s.d_model * s.d_ff  # GeGLU gate, up, down
    return proj + attn + mlp


def decoder_layer_flops(s: StackConfig, n_enc: float, m: float, i: int) -> float:
    """Merged-attention decoder layer [Eq. 11, flops_decoder]."""
    pairs = m * n_enc + (m * m if s.is_global(i) else m * min(m, s.window))
    proj = 4 * m * s.d_model * s.d_head * (s.n_heads + s.n_kv)
    enc_kv = 4 * n_enc * s.d_model * s.d_head * s.n_kv  # encoder-state K/V projections
    attn = 4 * pairs * s.n_heads * s.d_head
    mlp = 6 * m * s.d_model * s.d_ff
    return proj + enc_kv + attn + mlp


# --------------------------------------------------------------------------- #
# Whole-model costs
# --------------------------------------------------------------------------- #
def t5gemma2_flops(b: BackboneConfig, n: float, n_kept: float, m: float, k: int) -> float:
    """T5Gemma 2 with deletion after encoder layer ``k`` [Eq. 19].

    ``n_kept = n`` gives the unpruned cost [Eq. 15].
    """
    e, d = b.encoder, b.decoder
    assert d is not None
    enc = sum(layer_flops(e, n if i <= k else n_kept, i) for i in range(1, e.layers + 1))
    dec = sum(decoder_layer_flops(d, n_kept, m, i) for i in range(1, d.layers + 1))
    return enc + dec + 2 * m * d.d_model * b.vocab


def gemma3_flops(b: BackboneConfig, n_prompt: float, n_prompt_kept: float, m_gen: float, k: int) -> float:
    """Gemma 3 with prefill deletion after layer ``k`` [Eq. 20].

    Each layer sees ``T = N_prompt + M_gen`` (or the retained ``T~``) positions.
    """
    s = b.encoder
    t, t_kept = n_prompt + m_gen, n_prompt_kept + m_gen
    body = sum(layer_flops(s, t if i <= k else t_kept, i) for i in range(1, s.layers + 1))
    return body + 2 * m_gen * s.d_model * b.vocab


def selection_flops(b: BackboneConfig, method: str, n: float, n_visual: float, k: int) -> float:
    """Extra selection cost [Eq. 21, flops_gemma_dtm, flops_selectors].

    ``n`` is the full input (T5Gemma 2 encoder) or prompt (Gemma 3) length.
    """
    if n_visual <= 0 or method in ("base", "fastv", "random"):
        return 0.0
    s = b.encoder
    if method == "dtm":
        return 2 * n_visual * s.d_model  # one scalar projection per visual token
    if method == "phi_r":
        return 2 * self_pairs(s, n, k) * s.n_heads * s.d_head  # RoPE-free score contraction
    raise ValueError(f"unknown method {method!r}")


def example_flops(
    b: BackboneConfig, method: str, n_text: float, n_visual: float, n_visual_deleted: float, m: float, k: int = 5
) -> float:
    """Cost of one example: backbone at the retained length plus the selection cost."""
    n = n_text + n_visual
    n_kept = n - (0.0 if method == "base" else n_visual_deleted)
    if b.kind == "t5gemma2":
        backbone = t5gemma2_flops(b, n, n_kept, m, k)
    else:
        backbone = gemma3_flops(b, n, n_kept, m, k)
    return backbone + selection_flops(b, method, n, n_visual, k)


def flops_saved(
    b: BackboneConfig,
    method: str,
    examples: Iterable[Sequence[float]],
    k: int = 5,
) -> float:
    """Percentage of FLOPs saved, ``100 * (1 - sum F_method / sum F_base)``.

    ``examples`` yields ``(n_text, n_visual, n_visual_deleted, m_generated)``
    per evaluated example; the unpruned reference uses the same lengths.
    """
    num = den = 0.0
    for n_text, n_visual, n_deleted, m in examples:
        num += example_flops(b, method, n_text, n_visual, n_deleted, m, k)
        den += example_flops(b, "base", n_text, n_visual, 0.0, m, k)
    return 100.0 * (1.0 - num / den) if den else 0.0


__all__ = ["StackConfig", "BackboneConfig", "BACKBONES", "T5GEMMA2_1B", "T5GEMMA2_270M", "GEMMA3_4B",
           "layer_flops", "decoder_layer_flops", "t5gemma2_flops", "gemma3_flops", "selection_flops",
           "example_flops", "flops_saved"]
