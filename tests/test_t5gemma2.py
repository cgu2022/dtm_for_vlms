"""Tests for V-MrT5Gemma2 and the baselines on a tiny random T5Gemma 2 (CPU).

Tiny model: width 64, 8 encoder + 8 decoder layers (layer 6 is global, the
rest use a local window of 8), a one-layer SigLIP producing 25 visual tokens.
"""

import math

import pytest
import torch
from transformers.models.t5gemma2.configuration_t5gemma2 import (
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    T5Gemma2EncoderConfig,
)
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2ForConditionalGeneration

from dtm.config import DeletionConfig
from dtm.deletion import exact_budget_delete_mask
from dtm.models.t5gemma2 import SOFTMAX_1_ATTENTION, VMrT5Gemma2ForConditionalGeneration
from dtm.objective import observed_rate, total_loss

BOI, EOI, IMG, PAD, EOS, BOS = 97, 98, 99, 0, 1, 2
N_IMG = 25


def tiny_config() -> T5Gemma2Config:
    text = dict(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, query_pre_attn_scalar=32, sliding_window=8,
        pad_token_id=PAD, eos_token_id=EOS, bos_token_id=BOS,
    )
    vision = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        image_size=40, patch_size=4, num_channels=3,
    )
    encoder = T5Gemma2EncoderConfig(
        text_config=text, vision_config=vision, mm_tokens_per_image=N_IMG,
        boi_token_index=BOI, eoi_token_index=EOI, image_token_index=IMG,
    )
    return T5Gemma2Config(encoder=encoder, decoder=T5Gemma2DecoderConfig(**text), image_token_index=IMG)


def build(method: str, **kw) -> VMrT5Gemma2ForConditionalGeneration:
    torch.manual_seed(0)
    model = VMrT5Gemma2ForConditionalGeneration(tiny_config(), DeletionConfig(method=method, **kw))
    torch.nn.init.normal_(model.model.encoder.multi_modal_projector.mm_input_projection_weight, std=0.5)
    return model


def batch() -> dict[str, torch.Tensor]:
    """Two prompts ``<text> <boi> <img>*25 <eoi> <question>``; the second is right-padded."""
    g = torch.Generator().manual_seed(1)
    rows = [
        [BOS, 10, 11, 12, BOI] + [IMG] * N_IMG + [EOI] + list(range(20, 30)),
        [BOS, 13, BOI] + [IMG] * N_IMG + [EOI] + list(range(40, 46)),
    ]
    n = max(map(len, rows))
    ids = torch.tensor([r + [PAD] * (n - len(r)) for r in rows])
    return {
        "input_ids": ids,
        "attention_mask": (torch.arange(n)[None] < torch.tensor([[len(r)] for r in rows])).long(),
        "pixel_values": torch.randn(2, 3, 40, 40, generator=g),
    }


def labels() -> torch.Tensor:
    return torch.tensor([[50, 51, 52, EOS], [53, 54, EOS, -100]])


def n_text(b: dict) -> torch.Tensor:
    return ((b["input_ids"] != IMG) & b["attention_mask"].bool()).sum(1)


# --------------------------------------------------------------------------- #
def test_attention_backend_per_method():
    """softmax_1 for DTM only [Eq. 2]; every method runs eager attention."""
    for method, impl in (("dtm", SOFTMAX_1_ATTENTION), ("fastv", "eager"), ("base", "eager")):
        m = build(method)
        assert m.model.encoder.text_model.layers[7].self_attn.config._attn_implementation == impl
        assert m.model.decoder.layers[0].self_attn.config._attn_implementation == impl
        assert (m.gate is not None) == (method == "dtm")


def test_soft_forward_calibration_and_loss():
    """Soft training: calibration hits delta_init, g = 0 off-image, loss trains the gate."""
    model, b = build("dtm", delta=0.6), batch()
    model.train()
    model.calibrate_gate(**b, labels=labels(), delta_init=0.6)  # [App. A.2]
    out = model(**b, labels=labels())
    g, vis = out.gate_scores, out.visual_mask
    assert g.shape == b["input_ids"].shape and g.dtype == torch.float32
    assert torch.equal(vis.sum(1), torch.tensor([N_IMG, N_IMG]))
    assert torch.all(g[~vis] == 0) and torch.all((g[vis] < 0) & (g[vis] > -22))
    assert abs(observed_rate(g, vis, -7.0) - 0.6) <= 1.0 / (2 * N_IMG) + 1e-6
    loss = total_loss(out.loss, g, vis, alpha=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.gate.proj.weight.grad.abs().sum() > 0 and model.gate.proj.bias.grad.abs() > 0


@pytest.mark.parametrize("delta", [0.2, 0.6, 0.8])
def test_hard_deletion_exact_budget_keeps_text(delta):
    """Exact budget: floor(delta n_v) visual tokens removed; all text kept [Sec. 4.1, Sec. 5]."""
    model, b = build("dtm"), batch()
    model.eval()
    with torch.no_grad():
        enc = model.encode(**b, deletion_rate=delta, hard_delete=True)
    kept_visual = enc.key_visual.sum(1)
    kept_text = (enc.key_valid & ~enc.key_visual).sum(1)
    assert torch.equal(kept_visual, torch.full((2,), N_IMG - math.floor(delta * N_IMG)))
    assert torch.equal(kept_text, n_text(b))
    assert enc.last_hidden_state.shape[1] == int(enc.key_valid.sum(1).max())


def test_natural_mode_deletes_below_threshold_only():
    """Natural mode deletes g <= tau; text and boundary tokens are never deleted."""
    model, b = build("dtm"), batch()
    model.eval()
    with torch.no_grad():
        model.gate.proj.bias.fill_(-50.0)  # every visual g ~ c <= tau
        enc = model.encode(**b, hard_delete=True)
        assert enc.key_visual.sum() == 0 and torch.equal(enc.key_valid.sum(1), n_text(b))
        model.gate.proj.bias.fill_(50.0)  # every visual g ~ 0 > tau
        enc = model.encode(**b, hard_delete=True)
        assert torch.equal(enc.key_visual.sum(1), torch.tensor([N_IMG, N_IMG]))


def test_hard_deletion_equals_masking_in_original_positions():
    """Compaction keeps original positions and windows: removing tokens equals masking them as keys.

    Reference: the full sequence where, from layer k on, deleted keys get -inf and
    survivors keep g_j (encoder) / min(g_j - tau, 0) (decoder) [Sec. 4.1, Sec. 4.3].
    """
    model, b = build("dtm"), batch()
    model.eval()
    dec_ids = torch.tensor([[BOS, 50, 51], [BOS, 52, 53]])
    with torch.no_grad():
        hard = model(**b, decoder_input_ids=dec_ids, deletion_rate=0.6, hard_delete=True)
        enc_hard = model.encode(**b, deletion_rate=0.6, hard_delete=True)

        state = model._encoder_prefix(b["input_ids"], b["attention_mask"], b["pixel_values"], None, None)
        g = model.gate(state["hidden"], state["visual"])
        delete = exact_budget_delete_mask(g, state["visual"], 0.6)
        text = model.model.encoder.text_model
        h, pos = state["hidden"], state["position_ids"]
        neg = torch.finfo(h.dtype).min
        masks = model._encoder_masks(pos, state["valid"], torch.where(delete, torch.full_like(g, neg), g), h.dtype)
        rope = model._rope(h, pos)
        for i in range(5, 8):
            lt = text.config.layer_types[i]
            h = text.layers[i](h, rope[lt], masks[lt], pos)
        h = text.norm(h)
        keep = state["valid"] & ~delete
        for r in range(2):
            torch.testing.assert_close(enc_hard.last_hidden_state[r, : int(keep[r].sum())], h[r, keep[r]])

        ref_enc = type(enc_hard)(
            last_hidden_state=h, key_valid=keep, key_visual=state["visual"] & keep, key_gate=g,
            gate_scores=g, visual_mask=state["visual"],
        )
        ref = model(encoder_outputs=ref_enc, decoder_input_ids=dec_ids)
    torch.testing.assert_close(hard.logits, ref.logits)


def test_decoder_uses_threshold_shifted_bias():
    """Answer queries see min(g_j - tau, 0) on visual keys and 0 on text keys [Sec. 4.3]."""
    model, b = build("dtm"), batch()
    model.calibrate_gate(**b, delta_init=0.6)
    model.eval()
    dec_ids = torch.tensor([[BOS, 50, 51], [BOS, 52, 53]])
    with torch.no_grad():
        enc = model.encode(**b)  # soft
        mask = model._cross_mask(enc, 3, torch.float32)
        g, vis, valid = enc.key_gate, enc.key_visual, enc.key_valid
        expected = torch.where(vis, torch.clamp(g + 7.0, max=0.0), torch.zeros_like(g))
        assert mask.shape == (2, 1, 3, g.shape[1])
        torch.testing.assert_close(mask[:, 0, 0][valid], expected[valid])
        assert torch.all(mask[:, 0, 0][~valid] == torch.finfo(torch.float32).min)
        assert expected[vis].min() < -1.0  # some keys are suppressed
        with_bias = model(encoder_outputs=enc, decoder_input_ids=dec_ids).logits
        enc.key_gate = None
        without_bias = model(encoder_outputs=enc, decoder_input_ids=dec_ids).logits
    assert not torch.allclose(with_bias, without_bias)


def test_generate_matches_teacher_forcing():
    """generate() deletes once in the encoder; cached decoding agrees with a full forward."""
    model, b = build("dtm"), batch()
    model.eval()
    with torch.no_grad():
        out = model.generate(**b, deletion_rate=0.6, max_new_tokens=6, do_sample=False)
        assert out.shape[0] == 2 and 2 <= out.shape[1] <= 7
        logits = model(**b, decoder_input_ids=out[:, :-1], deletion_rate=0.6, hard_delete=True).logits
    pred = logits.argmax(-1)
    for r in range(2):
        seq = out[r, 1:].tolist()
        stop = seq.index(EOS) + 1 if EOS in seq else len(seq)
        assert pred[r, :stop].tolist() == seq[:stop]


@pytest.mark.parametrize("method", ["fastv", "phi_r", "random"])
def test_baselines_delete_exact_budget(method):
    """Baselines delete hard at the training rate while training and at any budget at inference [Sec. 5]."""
    model, b = build(method, delta=0.6), batch()
    model.train()
    enc = model.encode(**b)
    assert torch.equal(enc.key_visual.sum(1), torch.full((2,), N_IMG - math.floor(0.6 * N_IMG)))
    assert enc.gate_scores is None and enc.key_gate is None
    out = model(**b, labels=labels())
    out.loss.backward()  # trains the backbone
    model.eval()
    with torch.no_grad():
        for delta in (0.1, 0.4, 0.9):
            enc = model.encode(**b, deletion_rate=delta, hard_delete=True)
            assert torch.equal(enc.key_visual.sum(1), torch.full((2,), N_IMG - math.floor(delta * N_IMG)))
            assert torch.equal((enc.key_valid & ~enc.key_visual).sum(1), n_text(b))
        assert model.generate(**b, deletion_rate=0.4, max_new_tokens=3).shape[0] == 2


@pytest.mark.parametrize("aggregation", ["text_mean", "text_last"])
def test_fastv_query_aggregations(aggregation):
    model, b = build("fastv", aggregation=aggregation), batch()
    model.eval()
    with torch.no_grad():
        enc = model.encode(**b, deletion_rate=0.5, hard_delete=True)
    assert torch.equal(enc.key_visual.sum(1), torch.full((2,), N_IMG - math.floor(0.5 * N_IMG)))


def test_precomputed_image_embeddings_match_pixels():
    model, b = build("dtm"), batch()
    model.eval()
    with torch.no_grad():
        feats = model.get_image_features(b["pixel_values"], return_dict=True).pooler_output
        a = model(**b, decoder_input_ids=torch.tensor([[BOS], [BOS]]), deletion_rate=0.4, hard_delete=True)
        pre = {k: v for k, v in b.items() if k != "pixel_values"}
        c = model(**pre, image_embeds=feats, decoder_input_ids=torch.tensor([[BOS], [BOS]]),
                  deletion_rate=0.4, hard_delete=True)
    torch.testing.assert_close(a.logits, c.logits)


def test_base_matches_backbone_and_from_pretrained(tmp_path):
    """Base (no deletion) reproduces the unmodified backbone; from_pretrained_dtm attaches an fp32 gate."""
    torch.manual_seed(0)
    ref = T5Gemma2ForConditionalGeneration(tiny_config()).eval()
    ref.save_pretrained(tmp_path)
    b, dec_ids = batch(), torch.tensor([[BOS, 50], [BOS, 51]])
    base = VMrT5Gemma2ForConditionalGeneration.from_pretrained_dtm(str(tmp_path), DeletionConfig(method="base")).eval()
    with torch.no_grad():
        expected = ref(**b, decoder_input_ids=dec_ids).logits
        out = base(**b, decoder_input_ids=dec_ids)
    torch.testing.assert_close(out.logits, expected, atol=1e-5, rtol=1e-4)
    assert out.gate_scores is None

    dtm = VMrT5Gemma2ForConditionalGeneration.from_pretrained_dtm(
        str(tmp_path), DeletionConfig(method="dtm"), dtype=torch.bfloat16
    )
    assert dtm.gate.proj.weight.dtype == torch.float32
    assert dtm.gate.proj.weight.abs().sum() > 0 and dtm.gate.proj.bias.item() == 0.0
    assert dtm.model.decoder.layers[0].self_attn.config._attn_implementation == SOFTMAX_1_ATTENTION
