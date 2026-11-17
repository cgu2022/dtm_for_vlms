# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
# Modified for DTM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""V-MrGemma3: DTM in the decoder-only Gemma 3 (Sec. 4.2, Fig. 1b).

The prompt is ``[bos] <start_of_image> [256 image tokens] <end_of_image>
Question: {q}\\nAnswer:``. Gemma 3 attends bidirectionally inside an image
block and causally elsewhere, so the image tokens (and hence the gate) never
see the question that follows them (Sec. 4.2).

The HuggingFace ``Gemma3ForConditionalGeneration`` is extended with its own
language-model layer loop (the frozen SigLIP tower and projector are reused
unchanged):

1. **Prefill.** Layers ``1..k-1`` (``k = delete_after_layer + 1``; L5 means
   ``k = 6``) process and cache the full prompt. The states entering layer
   ``k`` are then scored:

   * DTM, soft (training): the gate (Eq. 1) scores the visual tokens (text and
     image-boundary tokens get ``g = 0``); in every later layer prompt queries
     add ``g_j`` to key ``j``'s logits and answer queries add
     ``min(g_j - tau, 0)`` for visual keys [Eq. 2, Sec. 4.3].
   * DTM, hard (inference): visual tokens with ``g_j <= tau`` (natural mode),
     or exactly ``floor(delta * n_v)`` lowest-``g`` visual tokens (exact
     budget), are removed. Survivors keep their biases [Sec. 4.3].
   * FastV / phi-R / Random: exactly ``floor(delta * n_v)`` visual tokens are
     removed, in training and at inference, with no gradient through the
     ranking [Sec. 5 "Baselines"]. FastV reuses layer ``k-1``'s attention
     weights, averaged over heads and prompt queries (``all_mean``); phi-R
     rescores layer ``k-1``'s normalised, pre-RoPE queries and keys.
   * Base: no deletion.

   Layers ``k..L`` then compute and cache only the compacted prompt.

2. **Masks in original position space** (Sec. 4.2, App. B.3). Survivors keep
   their original position ids; causal, local-window (``pos_i - pos_j <
   window``) and within-image bidirectional visibility are evaluated on those
   positions, so each surviving pair's visibility is unchanged.

3. **Decoding.** Generated tokens are never deleted and are appended to both
   cache regions; decode queries use the answer-query bias.

4. **Attention normaliser.** DTM models use softmax_1 in every language-model
   layer [Eq. 2]; all other methods use the standard softmax. All methods run
   eager attention so that masks can carry additive biases.

Layer indexing follows App. B.3: layer 0 is the input states and transformer
layers are numbered from 1, so with L5 the zero-based ``layers[0:5]`` see the
full prompt and ``layers[5:]`` (starting with global layer 6) the compacted one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Gemma3Config, Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3CausalLMOutputWithPast,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ..attention import answer_query_bias, eager_attention_forward_softmax_1
from ..baselines import fastv_scores, phi_r_scores, random_scores
from ..config import DeletionConfig
from ..deletion import compact, exact_budget_delete_mask, gather_seq, natural_delete_mask
from ..gate import DeleteGate

AttentionFn = Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]]


@dataclass
class DTMCausalLMOutput(Gemma3CausalLMOutputWithPast):
    """``Gemma3CausalLMOutputWithPast`` plus deletion diagnostics.

    Attributes:
        gate_scores: ``(B, N)`` DTM gate scores ``g`` over the input sequence
            (0 at non-visual tokens); None for the other methods.
        visual_mask: ``(B, N)`` visual tokens of the input (deletion candidates).
        kept_index: ``(B, N')`` original position of every row of ``logits``
            when hard deletion shortened the sequence (-1 on right padding of
            shorter rows); None when ``logits`` covers the full input.
    """

    gate_scores: Optional[torch.FloatTensor] = None
    visual_mask: Optional[torch.BoolTensor] = None
    kept_index: Optional[torch.LongTensor] = None


@dataclass
class TokenState:
    """Per-token metadata in original position space [Sec. 4.2].

    All tensors are ``(B, N)``. ``image`` is the image-block id (-1 outside
    images); ``answer`` marks queries that use the answer-query bias;
    ``gate`` holds ``g`` (0 at non-visual tokens) or None when no gate bias.
    """

    pos: torch.Tensor
    valid: torch.Tensor
    image: torch.Tensor
    visual: torch.Tensor
    answer: torch.Tensor
    gate: Optional[torch.Tensor] = None

    def gather(self, index: torch.Tensor, valid: torch.Tensor) -> "TokenState":
        """Survivors after compaction; ``valid`` marks the packed slots."""
        take = lambda x: gather_seq(x, index)  # noqa: E731
        gate = None if self.gate is None else take(self.gate)
        return TokenState(take(self.pos), valid, take(self.image), take(self.visual) & valid, take(self.answer), gate)

    def append(self, new: "TokenState") -> "TokenState":
        """Append generated tokens (never deleted, no gate bias) [Sec. 4.2]."""
        cat = lambda a, b: torch.cat([a, b], dim=1)  # noqa: E731
        gate = None if self.gate is None else cat(self.gate, torch.zeros_like(new.pos, dtype=self.gate.dtype))
        return TokenState(
            cat(self.pos, new.pos), cat(self.valid, new.valid), cat(self.image, new.image),
            cat(self.visual, new.visual), cat(self.answer, new.answer), gate,
        )


class PromptKVCache:
    """Key-value cache with two regions (Sec. 4.2, Fig. 1b).

    Layers ``1..k-1`` cache the full prompt (``regions[0]``) and layers
    ``k..L`` the compacted prompt (``regions[1]``); generated tokens are
    appended to both. Keys are stored after RoPE at their original positions.
    """

    def __init__(self, num_layers: int, split: int) -> None:
        self.keys: list[Optional[torch.Tensor]] = [None] * num_layers
        self.values: list[Optional[torch.Tensor]] = [None] * num_layers
        self.split = split
        self.regions: list[TokenState] = []

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer_idx] is not None:
            key = torch.cat([self.keys[layer_idx], key], dim=2)
            value = torch.cat([self.values[layer_idx], value], dim=2)
        self.keys[layer_idx], self.values[layer_idx] = key, value
        return key, value

    def seq_length(self, layer_idx: int) -> int:
        return 0 if self.keys[layer_idx] is None else self.keys[layer_idx].shape[2]


def build_attention_mask(
    queries: TokenState,
    keys: TokenState,
    sliding_window: Optional[int],
    gate_bias: bool,
    tau: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive ``(B, 1, Q, K)`` mask ``M_ij (+ gate bias)`` [Eq. 2, Sec. 4.2, App. B.3].

    Visibility is evaluated on original positions: causal, bidirectional within
    an image block, and ``pos_i - pos_j < window`` in local layers. With
    ``gate_bias``, prompt queries add ``g_j`` and answer queries add
    ``min(g_j - tau, 0)`` for visual keys (0 for text keys) [Sec. 4.3].
    """
    pq, pk = queries.pos[:, :, None], keys.pos[:, None, :]
    allowed = pk <= pq
    if sliding_window is not None:
        allowed = allowed & (pq - pk < sliding_window)
    same_image = (queries.image[:, :, None] >= 0) & (queries.image[:, :, None] == keys.image[:, None, :])
    allowed = (allowed | same_image) & keys.valid[:, None, :]
    bias = torch.zeros(allowed.shape, dtype=torch.float32, device=allowed.device)
    if gate_bias and keys.gate is not None:
        prompt_bias = keys.gate[:, None, :]  # g_j [Eq. 2]
        answer_bias = answer_query_bias(keys.gate, keys.visual, tau)[:, None, :]  # min(g_j-tau,0) [Sec. 4.3]
        bias = torch.where(queries.answer[:, :, None], answer_bias, prompt_bias)
    return torch.where(allowed, bias.to(dtype), torch.finfo(dtype).min)[:, None]


def _last_valid(valid: torch.Tensor) -> torch.Tensor:
    """Index of the last valid slot of each row."""
    idx = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    return torch.where(valid, idx, -1).amax(dim=1)


class VMrGemma3ForConditionalGeneration(Gemma3ForConditionalGeneration):
    """Gemma 3 with visual-token deletion during prefill (V-MrGemma3, Sec. 4.2).

    Args:
        config: a ``Gemma3Config``.
        deletion: method and settings; None means Base (no deletion).
    """

    def __init__(self, config: Gemma3Config, deletion: Optional[DeletionConfig] = None) -> None:
        super().__init__(config)
        self.set_deletion(deletion or DeletionConfig(method="base"))

    @classmethod
    def from_pretrained_dtm(
        cls, name_or_path: str, deletion: DeletionConfig, **hf_kwargs: Any
    ) -> "VMrGemma3ForConditionalGeneration":
        """Load pretrained Gemma 3 weights, then attach the deletion method (fp32 gate)."""
        model = cls.from_pretrained(name_or_path, **hf_kwargs)
        model.set_deletion(deletion)
        return model

    def set_deletion(self, deletion: DeletionConfig) -> None:
        num_layers = self.config.text_config.num_hidden_layers
        if not deletion.delete_after_layer < num_layers:
            raise ValueError(f"delete_after_layer must be < {num_layers}")
        self.deletion = deletion
        self._rng: Optional[torch.Generator] = None
        gate = None
        if deletion.method == "dtm":
            device = self.get_input_embeddings().weight.device
            gate = DeleteGate(self.config.text_config.hidden_size, c=deletion.c, tau=deletion.tau).to(device)
        self.gate = gate

    # ------------------------------------------------------------------ inputs
    def _embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        pixel_values: Optional[torch.FloatTensor],
        image_embeds: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor],
    ) -> tuple[torch.Tensor, TokenState]:
        """Token embeddings with image features merged in, plus token metadata."""
        if input_ids is None:
            raise ValueError("input_ids are required (visual tokens are located by the image token id)")
        image_token_id = self.config.image_token_id
        oov = image_token_id >= self.config.text_config.vocab_size
        embeds = self.get_input_embeddings()(input_ids.masked_fill(input_ids == image_token_id, 0) if oov else input_ids)
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.to(torch.bool)
        answer = torch.zeros_like(valid) if labels is None else (labels != -100) & valid
        is_image = (input_ids == image_token_id) & ~answer  # answer tokens are never visual [Sec. 4.2]
        if pixel_values is not None:  # frozen SigLIP tower + pooling + projector
            image_embeds = self.model.get_image_features(pixel_values, return_dict=True).pooler_output
        if image_embeds is not None:
            if image_embeds.numel() != int(is_image.sum()) * embeds.shape[-1]:
                raise ValueError("number of image features does not match the number of image tokens")
            embeds = embeds.masked_scatter(is_image[..., None].expand_as(embeds), image_embeds.to(embeds.dtype))
        if position_ids is None:
            position_ids = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        start = is_image & ~F.pad(is_image, (1, 0))[:, :-1]  # image-block starts
        image = torch.where(is_image, start.long().cumsum(dim=1) - 1, torch.full_like(position_ids, -1))
        tokens = TokenState(position_ids, valid, image, is_image & valid, answer)
        return embeds, tokens

    # ------------------------------------------------------------------ layers
    def _attention_fn(self) -> AttentionFn:
        return eager_attention_forward_softmax_1 if self.deletion.use_softmax_1 else eager_attention_forward  # [Eq. 2]

    def _self_attention(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        cache: Optional[PromptKVCache],
        capture: Optional[str],
    ) -> tuple[torch.Tensor, Any]:
        """``Gemma3Attention.forward`` with an explicit mask, cache and score capture."""
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(shape).transpose(1, 2))
        k = attn.k_norm(attn.k_proj(x).view(shape).transpose(1, 2))
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        captured: Any = (q.detach(), k.detach()) if capture == "qk" else None  # phi-R: pre-RoPE [Sec. 5]
        q, k = apply_rotary_pos_emb(q, k, *rope)
        if cache is not None:
            k, v = cache.update(k, v, attn.layer_idx)
        out, weights = self._attention_fn()(
            attn, q, k, v, mask, dropout=attn.attention_dropout if self.training else 0.0, scaling=attn.scaling
        )
        if capture == "weights":
            captured = weights.detach()  # FastV reuses them [Sec. 5]
        return attn.o_proj(out.reshape(*x.shape[:-1], -1).contiguous()), captured

    def _decoder_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        cache: Optional[PromptKVCache],
        capture: Optional[str],
    ) -> tuple[torch.Tensor, Any]:
        """``Gemma3DecoderLayer.forward`` around `_self_attention`."""
        attn_out, captured = self._self_attention(
            layer.self_attn, layer.input_layernorm(hidden), rope, mask, cache, capture
        )
        hidden = hidden + layer.post_attention_layernorm(attn_out)
        hidden = hidden + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(hidden)))
        return hidden, captured

    def _run_layers(
        self,
        hidden: torch.Tensor,
        queries: TokenState,
        keys: TokenState,
        layer_ids: range,
        cache: Optional[PromptKVCache],
        gate_bias: bool,
        capture: Optional[str] = None,
    ) -> tuple[torch.Tensor, Any]:
        """Run ``layers[layer_ids]``; ``capture`` records scores of the last layer of the range."""
        lm = self.model.language_model
        masks: dict[str, torch.Tensor] = {}
        ropes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        captured: Any = None
        for i in layer_ids:
            layer, layer_type = lm.layers[i], lm.config.layer_types[i]
            if layer_type not in masks:
                masks[layer_type] = build_attention_mask(
                    queries, keys, layer.self_attn.sliding_window, gate_bias, self.deletion.tau, hidden.dtype
                )
                ropes[layer_type] = lm.rotary_emb(hidden, queries.pos, layer_type)
            layer_capture = capture if i == layer_ids[-1] else None
            args = (layer, hidden, ropes[layer_type], masks[layer_type], cache, layer_capture)
            if self.training and layer.gradient_checkpointing and cache is None and layer_capture is None:
                hidden, _ = checkpoint(self._decoder_layer, *args, use_reentrant=False)
            else:
                hidden, out = self._decoder_layer(*args)
                if layer_capture is not None:
                    captured = (out, masks[layer_type])
        return hidden, captured

    # ------------------------------------------------------------------ deletion
    def _generator(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng.device != device:
            self._rng = torch.Generator(device=device).manual_seed(self.deletion.seed)
        return self._rng

    def _select(
        self, hidden: torch.Tensor, tokens: TokenState, captured: Any, hard: bool, rate: Optional[float]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Scores at the layer-``k`` boundary -> (delete mask or None, gate scores or None)."""
        d = self.deletion
        candidates = tokens.visual & tokens.valid  # visual tokens only [Sec. 4.1]
        if d.method == "base":
            return None, None
        if d.method == "dtm":
            g = self.gate(hidden, candidates)  # [Eq. 1]; g = 0 off the image
            if not hard:
                return None, g  # soft deletion [Eq. 2]
            if rate is None:
                return natural_delete_mask(g, candidates, d.tau), g  # g_j <= tau [Sec. 3.2]
            return exact_budget_delete_mask(g.detach(), candidates, rate), g  # floor(delta n_v) [Sec. 5]
        with torch.no_grad():  # fixed ranking rules [Sec. 5]
            queries = tokens.valid & ~tokens.answer  # prompt queries (Gemma 3)
            text = queries & ~tokens.visual
            if d.method == "fastv":
                scores = fastv_scores(captured[0], queries, text, d.aggregation)
            elif d.method == "phi_r":
                (q, k), mask = captured
                scaling = self.model.language_model.layers[d.delete_after_layer - 1].self_attn.scaling
                scores = phi_r_scores(q, k, mask, scaling, queries, text, d.aggregation)
            else:
                scores = random_scores(candidates, self._generator(hidden.device))
        return exact_budget_delete_mask(scores, candidates, d.delta if rate is None else rate), None

    def _prompt_forward(
        self,
        embeds: torch.Tensor,
        tokens: TokenState,
        hard: bool,
        rate: Optional[float],
        cache: Optional[PromptKVCache],
    ) -> tuple[torch.Tensor, TokenState, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Full-sequence pass with deletion after layer ``k-1`` [Sec. 4.2].

        Returns the final-normed hidden states, the metadata of the (possibly
        compacted) sequence, the gate scores, and the compaction index.
        """
        split = self.deletion.delete_after_layer
        num_layers = self.config.text_config.num_hidden_layers
        capture = {"fastv": "weights", "phi_r": "qk"}.get(self.deletion.method)
        hidden, captured = self._run_layers(embeds, tokens, tokens, range(split), cache, False, capture)
        delete, g = self._select(hidden, tokens, captured, hard, rate)
        full = tokens
        if g is not None:
            tokens = replace(tokens, gate=g)
        index = None
        if delete is not None:  # hard deletion: compact [Sec. 4.2]
            hidden, index, valid = compact(hidden, tokens.valid & ~delete)
            tokens = tokens.gather(index, valid)
            index = index.masked_fill(~valid, -1)
        if cache is not None:
            cache.regions = [full, tokens]
        hidden, _ = self._run_layers(hidden, tokens, tokens, range(split, num_layers), cache, True)
        return self.model.language_model.norm(hidden), tokens, g, index

    def _hard(self, hard_delete: bool) -> bool:
        """Baselines always delete hard; DTM only when asked [Sec. 3.2, 5]."""
        return hard_delete or self.deletion.method != "dtm"

    # ------------------------------------------------------------------ API
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        deletion_rate: Optional[float] = None,
        hard_delete: bool = False,
        **hf_kwargs: Any,
    ) -> DTMCausalLMOutput:
        """Teacher-forced pass over prompt and answer.

        Args:
            labels: next-token targets in HF convention (-100 on the prompt);
                tokens with a label are answer queries [Sec. 4.3].
            image_embeds: precomputed projected image features, used instead
                of ``pixel_values``.
            deletion_rate: exact-budget rate; baselines default to
                ``deletion.delta``; DTM with ``hard_delete`` and no rate uses
                the natural threshold ``g <= tau``.
            hard_delete: DTM hard deletion (inference); otherwise DTM deletes
                softly. Baselines always delete hard.
            hf_kwargs: accepted for interface compatibility (e.g.
                ``token_type_ids``, ``use_cache``, ``return_dict``); cached
                decoding goes through :meth:`generate`.

        When hard deletion shortens the sequence, ``logits`` and the loss are
        computed on the survivors (see ``kept_index``).
        """
        if hf_kwargs.get("past_key_values") is not None:
            raise ValueError("cached decoding is provided by generate()")
        embeds, tokens = self._embed(input_ids, attention_mask, position_ids, pixel_values, image_embeds, labels)
        hidden, kept, g, index = self._prompt_forward(
            embeds, tokens, self._hard(hard_delete), deletion_rate, cache=None
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            targets = F.pad(labels.masked_fill(~tokens.valid, -100)[:, 1:], (0, 1), value=-100)  # next token
            if index is not None:
                targets = gather_seq(targets, index.clamp_min(0)).masked_fill(~kept.valid, -100)
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
        return DTMCausalLMOutput(
            loss=loss, logits=logits, gate_scores=g, visual_mask=tokens.visual, kept_index=index
        )

    @torch.no_grad()
    def calibrate_gate(self, *, delta_init: Optional[float] = None, **batch: Any) -> None:
        """One-time gate-bias calibration on the first training batch [Sec. 4.3, App. A.2].

        Runs layers ``1..k-1`` on ``batch`` and sets the gate bias so that a
        fraction ``delta_init`` (default ``deletion.calibration_rate``) of the
        batch's visual tokens starts at ``g <= tau``.
        """
        if self.gate is None:
            raise ValueError("calibrate_gate requires method='dtm'")
        embeds, tokens = self._embed(
            batch["input_ids"], batch.get("attention_mask"), batch.get("position_ids"),
            batch.get("pixel_values"), batch.get("image_embeds"), None,
        )
        split = self.deletion.delete_after_layer
        hidden, _ = self._run_layers(embeds, tokens, tokens, range(split), None, False)
        rate = self.deletion.calibration_rate if delta_init is None else delta_init
        self.gate.calibrate_(hidden, tokens.visual & tokens.valid, rate)

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        deletion_rate: Optional[float] = None,
        hard_delete: bool = True,
    ) -> tuple[torch.Tensor, PromptKVCache, torch.Tensor]:
        """Prompt prefill with deletion [Sec. 4.2].

        Returns the next-token logits ``(B, V)``, the two-region cache, and the
        position id of the next token ``(B,)``.
        """
        num_layers = self.config.text_config.num_hidden_layers
        cache = PromptKVCache(num_layers, self.deletion.delete_after_layer)
        embeds, tokens = self._embed(input_ids, attention_mask, None, pixel_values, image_embeds, None)
        hidden, kept, _, _ = self._prompt_forward(embeds, tokens, self._hard(hard_delete), deletion_rate, cache)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        logits = self.lm_head(hidden[rows, _last_valid(kept.valid)])
        next_pos = tokens.pos.masked_fill(~tokens.valid, -1).amax(dim=1) + 1
        return logits, cache, next_pos

    @torch.no_grad()
    def decode_step(self, token: torch.LongTensor, pos: torch.LongTensor, cache: PromptKVCache) -> torch.Tensor:
        """One cached decoding step for ``token`` ``(B,)`` at position ``pos`` ``(B,)``; returns logits ``(B, V)``.

        The new token is appended to both cache regions and never deleted; as
        an answer query it sees ``min(g_j - tau, 0)`` on surviving visual keys
        in layers ``k..L`` [Sec. 4.2, 4.3].
        """
        ones = torch.ones_like(token, dtype=torch.bool)[:, None]
        new = TokenState(pos[:, None], ones, torch.full_like(pos, -1)[:, None], ~ones, ones)
        cache.regions = [region.append(new) for region in cache.regions]
        ids = token[:, None]
        if self.config.image_token_id >= self.config.text_config.vocab_size:
            ids = ids.masked_fill(ids == self.config.image_token_id, 0)
        hidden = self.get_input_embeddings()(ids)
        split, num_layers = cache.split, self.config.text_config.num_hidden_layers
        hidden, _ = self._run_layers(hidden, new, cache.regions[0], range(split), cache, False)
        hidden, _ = self._run_layers(hidden, new, cache.regions[1], range(split, num_layers), cache, True)
        return self.lm_head(self.model.language_model.norm(hidden[:, -1]))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        deletion_rate: Optional[float] = None,
        hard_delete: bool = True,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        eos_token_id: Optional[Union[int, list[int]]] = None,
        pad_token_id: Optional[int] = None,
        **hf_kwargs: Any,
    ) -> torch.LongTensor:
        """Greedy decoding with deletion applied once, during prefill [Sec. 4.2].

        Returns ``input_ids`` followed by the generated tokens; rows that have
        emitted an EOS token are continued with ``pad_token_id``. Other
        ``GenerationMixin.generate`` keyword arguments are ignored.
        """
        if do_sample:
            raise NotImplementedError("only greedy decoding is implemented")
        if eos_token_id is None:
            eos_token_id = self.generation_config.eos_token_id
        eos = torch.tensor(
            [] if eos_token_id is None else [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id),
            dtype=torch.long, device=input_ids.device,
        )
        if pad_token_id is None:
            pad_token_id = self.config.text_config.pad_token_id or 0
        logits, cache, pos = self.prefill(
            input_ids, attention_mask, pixel_values, image_embeds, deletion_rate, hard_delete
        )
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        generated: list[torch.Tensor] = []
        for step in range(max_new_tokens):
            token = torch.where(done, torch.full_like(pos, pad_token_id), logits.argmax(dim=-1))
            generated.append(token)
            done = done | torch.isin(token, eos)
            if bool(done.all()) or step == max_new_tokens - 1:
                break
            logits = self.decode_step(token, pos, cache)
            pos = pos + 1
        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1) if generated else input_ids


__all__ = [
    "VMrGemma3ForConditionalGeneration",
    "DTMCausalLMOutput",
    "PromptKVCache",
    "TokenState",
    "build_attention_mask",
]
