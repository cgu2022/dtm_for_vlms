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
"""V-MrT5Gemma2: DTM on the T5Gemma 2 encoder--decoder (Sec. 4.1, Fig. 1a).

The HuggingFace ``T5Gemma2ForConditionalGeneration`` is extended in four places:

1. **Encoder layer loop.** Layers ``1..k-1`` (``k = delete_after_layer + 1``;
   L5 means ``k = 6``) process the full sequence. The states entering layer
   ``k`` are then scored and, depending on the method:

   * DTM, soft (training): the delete gate (Eq. 1) scores the visual tokens
     (text and image-boundary tokens get ``g = 0``), and ``g_j`` is added to key
     ``j``'s logits in every later encoder layer [Eq. 2].
   * DTM, hard (inference): tokens with ``g_j <= tau`` (natural mode), or
     exactly ``floor(delta * n_v)`` lowest-``g`` visual tokens (exact budget),
     are removed. Survivors keep their ``g_j`` as a key bias [Sec. 4.3].
   * FastV / phi-R / Random: exactly ``floor(delta * n_v)`` visual tokens are
     removed, in training and at inference [Sec. 5 "Baselines"]. FastV reuses
     the attention weights of layer ``k-1`` (eager attention); phi-R rescores
     layer ``k-1``'s normalised, pre-RoPE queries and keys [App. D.6].

   Compaction keeps each survivor's original position id, and the attention
   masks of later layers are rebuilt from those original positions, so every
   surviving pair keeps its original local-window visibility [Sec. 4.1,
   App. B.3].

2. **Decoder merged attention** (App. B.3, Eq. 8). The encoder-key block of
   the merged mask carries ``min(g_j - tau, 0)`` for visual keys and ``0``
   for text keys [Sec. 4.3]. After hard deletion the
   decoder reads the shortened encoder states in every layer.

3. **Attention normaliser.** DTM models use softmax_1 in every attention layer
   of both stacks [Eq. 2]; all other methods use the standard softmax. All
   methods run eager attention so that masks can carry additive biases.

4. **Generation.** Deletion is applied once, when the encoder runs; cached
   decoding then reads the compacted encoder output.

Layer indexing follows App. B.3: layer 0 is the input states and transformer
layers are numbered from 1, so with L5 the zero-based ``layers[0:5]`` see the
full sequence and ``layers[5:]`` (starting with global layer 6) the shortened one.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import AttentionInterface
from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2Config
from transformers.models.t5gemma2.modeling_t5gemma2 import (
    T5Gemma2EncoderLayer,
    T5Gemma2ForConditionalGeneration,
)

from ..attention import add_key_bias, answer_query_bias, eager_attention_forward_softmax_1
from ..baselines import fastv_scores, phi_r_scores, random_scores
from ..config import DeletionConfig
from ..deletion import compact, exact_budget_delete_mask, gather_seq, natural_delete_mask
from ..gate import DeleteGate

# softmax_1 attention backend [Eq. 2]; masks are built exactly as for eager attention.
SOFTMAX_1_ATTENTION = "dtm_eager_softmax_1"
AttentionInterface.register(SOFTMAX_1_ATTENTION, eager_attention_forward_softmax_1)
AttentionMaskInterface.register(SOFTMAX_1_ATTENTION, eager_mask)


@dataclass
class DTMEncoderOutput(BaseModelOutput):
    """Encoder output after deletion.

    ``last_hidden_state``, ``key_valid``, ``key_visual`` and ``key_gate`` are
    aligned with the (possibly shortened) sequence the decoder reads;
    ``gate_scores`` and ``visual_mask`` are aligned with the full input.
    """

    key_valid: Optional[torch.Tensor] = None
    key_visual: Optional[torch.Tensor] = None
    key_gate: Optional[torch.Tensor] = None
    gate_scores: Optional[torch.Tensor] = None
    visual_mask: Optional[torch.Tensor] = None


@dataclass
class DTMSeq2SeqLMOutput(Seq2SeqLMOutput):
    """``Seq2SeqLMOutput`` plus the gate scores ``g`` and visual mask over the full input."""

    gate_scores: Optional[torch.Tensor] = None
    visual_mask: Optional[torch.Tensor] = None


class VMrT5Gemma2ForConditionalGeneration(T5Gemma2ForConditionalGeneration):
    """T5Gemma 2 with visual token deletion in the encoder (V-MrT5Gemma2 and baselines).

    Args:
        config: a ``T5Gemma2Config``.
        deletion: the deletion method and its settings.
    """

    _keep_in_fp32_modules_strict = ["gate"]

    def __init__(self, config: T5Gemma2Config, deletion: Optional[DeletionConfig] = None) -> None:
        super().__init__(config)
        self.set_deletion(deletion if deletion is not None else DeletionConfig())

    @classmethod
    def from_pretrained_dtm(
        cls, name_or_path: str, deletion: DeletionConfig, **hf_kwargs: Any
    ) -> "VMrT5Gemma2ForConditionalGeneration":
        """Load a pretrained T5Gemma 2 checkpoint and attach the deletion method.

        The gate is freshly initialised (Eq. 1) and kept in fp32; its bias is
        set later by :meth:`calibrate_gate` [App. A.2].
        """
        model = cls.from_pretrained(name_or_path, **hf_kwargs)
        model.set_deletion(deletion)
        return model

    def set_deletion(self, deletion: DeletionConfig) -> None:
        """Attach a deletion method: a new fp32 gate for DTM, no gate otherwise."""
        n_layers = self.config.encoder.text_config.num_hidden_layers
        if deletion.delete_after_layer >= n_layers:
            raise ValueError(f"delete_after_layer must be < {n_layers}")
        self.deletion = deletion
        self.gate: Optional[DeleteGate] = None
        if deletion.method == "dtm":
            device = self.get_input_embeddings().weight.device
            gate = DeleteGate(self.config.encoder.text_config.hidden_size, c=deletion.c, tau=deletion.tau)
            self.gate = gate.to(device=device, dtype=torch.float32)
        self._generator: Optional[torch.Generator] = None
        self._set_attention_backend()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _set_attention_backend(self) -> None:
        """softmax_1 for DTM, standard softmax otherwise; eager everywhere [Eq. 2]."""
        impl = SOFTMAX_1_ATTENTION if self.deletion.use_softmax_1 else "eager"
        stacks = (self.model.encoder.text_model, self.model.decoder)
        configs = {id(c): c for c in (self.config.encoder.text_config, self.config.decoder)}
        for stack in stacks:
            for module in stack.modules():
                cfg = getattr(module, "config", None)
                if cfg is not None and hasattr(cfg, "layer_types"):
                    configs[id(cfg)] = cfg
        for cfg in configs.values():
            cfg._attn_implementation = impl

    def _random_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device).manual_seed(self.deletion.seed)
        return self._generator

    # ------------------------------------------------------------------ #
    # Encoder
    # ------------------------------------------------------------------ #
    def _embed(
        self, input_ids: torch.LongTensor, pixel_values: Optional[torch.Tensor], image_embeds: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Token embeddings with the projected image features at the image placeholders.

        ``image_embeds`` (``(num_images, tokens_per_image, d)``, already pooled and
        projected) may replace ``pixel_values``.
        """
        encoder = self.model.encoder
        embeds = encoder.text_model.embed_tokens(input_ids)
        if image_embeds is None and pixel_values is not None:
            image_embeds = encoder.get_image_features(pixel_values, return_dict=True).pooler_output
        if image_embeds is not None:
            placeholder = (input_ids == self.config.encoder.image_token_id)[..., None].expand_as(embeds)
            embeds = embeds.masked_scatter(placeholder, image_embeds.to(embeds.device, embeds.dtype))
        return embeds

    def _encoder_masks(
        self, position_ids: torch.Tensor, valid: torch.Tensor, key_gate: Optional[torch.Tensor], dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Additive ``(B, 1, N, N)`` masks for global and local encoder layers.

        Visibility is computed from original position ids, so after compaction
        each surviving pair keeps its original bidirectional window: ``(W+1)//2``
        positions back (incl. itself) and ``W//2`` ahead [App. B.3]. ``key_gate``
        adds ``g_j`` to key ``j`` [Eq. 2].
        """
        window = self.config.encoder.text_config.sliding_window
        dist = position_ids[:, :, None] - position_ids[:, None, :]  # query - key
        local = ((dist >= 0) & (dist < (window + 1) // 2)) | ((dist < 0) & (-dist < window // 2 + 1))
        key_ok = valid[:, None, :]
        masks = {}
        for layer_type, visible in (("full_attention", key_ok.expand_as(local)), ("sliding_attention", local & key_ok)):
            mask = torch.zeros(visible.shape, dtype=dtype, device=visible.device)
            mask = mask.masked_fill(~visible, torch.finfo(dtype).min)[:, None]
            masks[layer_type] = mask if key_gate is None else add_key_bias(mask, key_gate, dtype)
        return masks

    def _rope(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
        text = self.model.encoder.text_model
        return {t: text.rotary_emb(hidden_states, position_ids, t) for t in set(text.config.layer_types)}

    @contextmanager
    def _capture_scoring_layer(self, layer: T5Gemma2EncoderLayer) -> Iterator[dict[str, torch.Tensor]]:
        """Record what FastV / phi-R need from layer ``k-1`` while it runs [Sec. 5]."""
        store: dict[str, torch.Tensor] = {}
        attn = layer.self_attn
        hooks = []
        if self.deletion.method == "fastv":  # the layer's own (eager) attention weights
            hooks.append(attn.register_forward_hook(lambda m, i, out: store.__setitem__("attn", out[1].detach())))
        elif self.deletion.method == "phi_r":  # normalised q/k before the rotary transform
            hooks.append(attn.q_norm.register_forward_hook(lambda m, i, out: store.__setitem__("q", out.detach())))
            hooks.append(attn.k_norm.register_forward_hook(lambda m, i, out: store.__setitem__("k", out.detach())))
        try:
            yield store
        finally:
            for hook in hooks:
                hook.remove()

    def _encoder_prefix(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        image_embeds: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
    ) -> dict[str, Any]:
        """Run encoder layers ``1..k-1`` on the full sequence."""
        text = self.model.encoder.text_model
        embeds = self._embed(input_ids, pixel_values, image_embeds)
        batch, length = input_ids.shape
        valid = (
            attention_mask.to(torch.bool)
            if attention_mask is not None
            else torch.ones(batch, length, dtype=torch.bool, device=input_ids.device)
        )
        # Only image placeholders are candidates; text and <boi>/<eoi> markers are not [Sec. 4.1].
        visual = (input_ids == self.config.encoder.image_token_id) & valid
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device)
        position_ids = position_ids.expand(batch, length)

        masks = self._encoder_masks(position_ids, valid, None, embeds.dtype)
        rope = self._rope(embeds, position_ids)
        types = text.config.layer_types
        n_full = self.deletion.delete_after_layer
        hidden = text.dropout(embeds)
        with self._capture_scoring_layer(text.layers[n_full - 1]) as store:
            for i in range(n_full):
                hidden = text.layers[i](hidden, rope[types[i]], masks[types[i]], position_ids)
        return {
            "hidden": hidden,
            "position_ids": position_ids,
            "valid": valid,
            "visual": visual,
            "store": store,
            "scoring_mask": masks[types[n_full - 1]],
        }

    def _selection_scores(self, state: dict[str, Any]) -> torch.Tensor:
        """Baseline ranking scores, higher = keep (no gradient) [Sec. 5, App. D.6]."""
        method, agg = self.deletion.method, self.deletion.aggregation
        valid, visual, store = state["valid"], state["visual"], state["store"]
        text_query = valid & ~visual
        if method == "fastv":
            return fastv_scores(store["attn"], valid, text_query, agg)
        if method == "phi_r":
            scoring_layer = self.model.encoder.text_model.layers[self.deletion.delete_after_layer - 1]
            return phi_r_scores(
                store["q"], store["k"], state["scoring_mask"], scoring_layer.self_attn.scaling, valid, text_query, agg
            )
        return random_scores(visual, self._random_generator(visual.device))

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        deletion_rate: Optional[float] = None,
        hard_delete: bool = False,
    ) -> DTMEncoderOutput:
        """Run the encoder with deletion after layer ``k-1``.

        * DTM: soft unless ``hard_delete``; hard deletion uses the exact budget
          ``floor(deletion_rate * n_v)`` when ``deletion_rate`` is given, else the
          natural threshold ``g <= tau``.
        * FastV / phi-R / Random: always hard, exact budget at ``deletion_rate``
          (default ``deletion.delta``).
        * Base: no deletion.
        """
        state = self._encoder_prefix(input_ids, attention_mask, pixel_values, image_embeds, position_ids)
        hidden, position_ids = state["hidden"], state["position_ids"]
        valid, visual = state["valid"], state["visual"]
        full_visual = visual
        method = self.deletion.method

        gate_scores: Optional[torch.Tensor] = None
        delete: Optional[torch.Tensor] = None
        if method == "dtm":
            gate_scores = self.gate(hidden, visual)  # [Eq. 1]
            if hard_delete:
                delete = (
                    exact_budget_delete_mask(gate_scores, visual, deletion_rate)  # [Sec. 5]
                    if deletion_rate is not None
                    else natural_delete_mask(gate_scores, visual, self.deletion.tau)  # [Sec. 3.2]
                )
        elif method != "base":
            rate = self.deletion.delta if deletion_rate is None else deletion_rate
            delete = exact_budget_delete_mask(self._selection_scores(state), visual, rate)

        key_gate = gate_scores
        if delete is not None:  # hard deletion: compact, keeping original positions [Sec. 4.1]
            hidden, index, valid = compact(hidden, valid & ~delete)
            position_ids = gather_seq(position_ids, index)
            visual = gather_seq(visual, index) & valid
            if key_gate is not None:
                key_gate = torch.where(valid, gather_seq(key_gate, index), torch.zeros_like(key_gate[:, :1]))

        # Layers k..L: encoder queries see g_j on visual keys (soft, or survivors when hard) [Eq. 2, Sec. 4.3].
        text = self.model.encoder.text_model
        masks = self._encoder_masks(position_ids, valid, key_gate, hidden.dtype)
        rope = self._rope(hidden, position_ids)
        types = text.config.layer_types
        for i in range(self.deletion.delete_after_layer, text.config.num_hidden_layers):
            hidden = text.layers[i](hidden, rope[types[i]], masks[types[i]], position_ids)
        hidden = text.dropout(text.norm(hidden))

        return DTMEncoderOutput(
            last_hidden_state=hidden,
            key_valid=valid,
            key_visual=visual,
            key_gate=key_gate,
            gate_scores=gate_scores,
            visual_mask=full_visual,
        )

    @torch.no_grad()
    def calibrate_gate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        delta_init: Optional[float] = None,
        **_: Any,
    ) -> None:
        """One-time gate-bias calibration on the first training batch [App. A.2, Eq. 6].

        ``delta_init`` defaults to ``deletion.calibration_rate``. Other batch
        entries (labels, decoder inputs) are ignored.
        """
        if self.gate is None:
            raise RuntimeError("calibrate_gate requires method='dtm'")
        state = self._encoder_prefix(input_ids, attention_mask, pixel_values, image_embeds, position_ids)
        rate = self.deletion.calibration_rate if delta_init is None else delta_init
        self.gate.calibrate_(state["hidden"], state["visual"], rate)

    # ------------------------------------------------------------------ #
    # Decoder
    # ------------------------------------------------------------------ #
    def _cross_mask(self, encoder_outputs: DTMEncoderOutput, query_length: int, dtype: torch.dtype) -> torch.Tensor:
        """Encoder-key block of the merged mask, ``(B, 1, Q, N')`` [App. B.3, Sec. 4.3].

        Padding keys are masked; with DTM, visual keys carry ``min(g_j - tau, 0)``
        and text keys ``0``.
        """
        valid = encoder_outputs.key_valid
        mask = torch.zeros(valid.shape, dtype=dtype, device=valid.device).masked_fill(~valid, torch.finfo(dtype).min)
        if encoder_outputs.key_gate is not None:
            mask = mask + answer_query_bias(encoder_outputs.key_gate, encoder_outputs.key_visual, self.deletion.tau).to(
                dtype
            )
        return mask[:, None, None, :].expand(-1, 1, query_length, -1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[DTMEncoderOutput] = None,
        past_key_values: Optional[Any] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_embeds: Optional[torch.Tensor] = None,
        deletion_rate: Optional[float] = None,
        hard_delete: bool = False,
        **kwargs: Any,
    ) -> DTMSeq2SeqLMOutput:
        """Seq2seq forward with visual token deletion.

        Args (beyond HuggingFace's):
            image_embeds: projected image features, instead of ``pixel_values``.
            deletion_rate: exact inference budget ``delta`` (all methods); the
                training rate of the baselines (default ``deletion.delta``).
                ``None`` with DTM and ``hard_delete`` selects the natural mode.
            hard_delete: physically remove tokens (DTM); training uses soft deletion.
        """
        if encoder_outputs is None:
            if input_ids is None:
                raise ValueError("input_ids are required to locate the visual tokens")
            encoder_outputs = self.encode(
                input_ids, attention_mask, pixel_values, image_embeds, position_ids, deletion_rate, hard_delete
            )
        elif not isinstance(encoder_outputs, DTMEncoderOutput):
            raise TypeError("encoder_outputs must come from VMrT5Gemma2ForConditionalGeneration.encode")

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)

        decoder = self.model.decoder
        query_length = (decoder_input_ids if decoder_input_ids is not None else decoder_inputs_embeds).shape[1]
        cross_mask = self._cross_mask(encoder_outputs, query_length, decoder.embed_tokens.weight.dtype)
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask={"full_attention": cross_mask},
            use_cache=use_cache,
            return_dict=True,
        )

        hidden_states = decoder_outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        softcap = self.config.decoder.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return DTMSeq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            gate_scores=encoder_outputs.gate_scores,
            visual_mask=encoder_outputs.visual_mask,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs: dict[str, Any],
        model_input_name: Optional[str],
        generation_config: Any,
    ) -> dict[str, Any]:
        """Run the encoder once, with deletion; decoding steps reuse its output."""
        model_kwargs["encoder_outputs"] = self.encode(
            inputs_tensor,
            attention_mask=model_kwargs.get("attention_mask"),
            pixel_values=model_kwargs.get("pixel_values"),
            image_embeds=model_kwargs.get("image_embeds"),
            position_ids=model_kwargs.get("position_ids"),
            deletion_rate=model_kwargs.get("deletion_rate"),
            hard_delete=model_kwargs.get("hard_delete", True),
        )
        return model_kwargs

    def generate(self, *args: Any, hard_delete: bool = True, **kwargs: Any) -> Any:
        """``GenerationMixin.generate`` with hard deletion by default (inference) [Sec. 3.2]."""
        return super().generate(*args, hard_delete=hard_delete, **kwargs)


__all__ = ["VMrT5Gemma2ForConditionalGeneration", "DTMEncoderOutput", "DTMSeq2SeqLMOutput", "SOFTMAX_1_ATTENTION"]
