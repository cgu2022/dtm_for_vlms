"""V-MrGemma3 (Sec. 4.2, Fig. 1b) on a tiny random Gemma 3, CPU only.

Checks soft deletion in training, KV-cache compaction after layer 5, cached
decoding against uncached greedy recomputation, and exact deletion budgets.
"""

import math

import pytest
import torch
from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig

from dtm.config import DeletionConfig
from dtm.models.gemma3 import VMrGemma3ForConditionalGeneration
from dtm.objective import gate_loss, observed_rate, total_loss

BOI, EOI, IMG = 100, 101, 102
N_IMAGE = 16  # visual tokens per image
DELTA = 0.6
SPLIT = 5  # L5: gate after layer 5, k = 6


def tiny_config() -> Gemma3Config:
    text = Gemma3TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, query_pre_attn_scalar=8,
        sliding_window=12, max_position_embeddings=256, initializer_range=0.2,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
    )  # layer_types: layers[5] (layer 6) is global, the rest local
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        image_size=32, patch_size=4,
    )
    return Gemma3Config(
        text_config=text, vision_config=vision, mm_tokens_per_image=N_IMAGE,
        boi_token_index=BOI, eoi_token_index=EOI, image_token_index=IMG, initializer_range=0.2,
    )


def make_model(method: str, **kwargs) -> VMrGemma3ForConditionalGeneration:
    torch.manual_seed(0)
    deletion = DeletionConfig(method=method, delete_after_layer=SPLIT, delta=DELTA, **kwargs)
    model = VMrGemma3ForConditionalGeneration(tiny_config(), deletion)
    torch.nn.init.normal_(model.model.multi_modal_projector.mm_input_projection_weight, std=0.5)
    return model


def prompt(batch: int = 2, question: int = 6, seed: int = 0) -> dict:
    """``[bos] <boi> [image] <eoi> question`` with random question tokens."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(3, 99, (batch, question), generator=g)
    head = torch.tensor([2, BOI] + [IMG] * N_IMAGE + [EOI]).expand(batch, -1)
    return {
        "input_ids": torch.cat([head, q], dim=1),
        "image_embeds": torch.randn(batch, N_IMAGE, 32, generator=g),
    }


def with_answer(batch: dict, answer_len: int = 3, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    answer = torch.randint(3, 99, (batch["input_ids"].shape[0], answer_len), generator=g)
    ids = torch.cat([batch["input_ids"], answer], dim=1)
    labels = torch.cat([torch.full_like(batch["input_ids"], -100), answer], dim=1)
    return {**batch, "input_ids": ids, "labels": labels}


def test_soft_forward_and_loss():
    """Training: soft deletion keeps the sequence intact and trains the gate (Eq. 1-2, Sec. 3.2)."""
    model = make_model("dtm").train()
    batch = with_answer(prompt())
    model.calibrate_gate(**batch, delta_init=0.5)  # App. A.2
    out = model(**batch)
    b, n = batch["input_ids"].shape
    assert out.logits.shape == (b, n, 128) and out.kept_index is None
    visual = batch["input_ids"] == IMG
    assert torch.equal(out.visual_mask, visual)
    g = out.gate_scores
    assert torch.all(g[~visual] == 0) and torch.all((g[visual] < 0) & (g[visual] > -22))
    assert abs(observed_rate(g, visual, -7.0) - 0.5) <= 1.0 / visual.sum().item() + 1e-6
    loss = total_loss(out.loss, g, visual, alpha=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.gate.proj.weight.grad.abs().sum() > 0
    assert torch.isclose(gate_loss(g, visual), g[visual].mean())


def test_soft_bias_uses_softmax_1_and_answer_shift():
    """Soft deletion changes attention only in layers >= k; answer queries see min(g - tau, 0) (Sec. 4.3)."""
    model = make_model("dtm").eval()
    batch = with_answer(prompt())
    with torch.no_grad():
        model.gate.proj.bias.fill_(-40.0)  # g ~ 0: nothing suppressed
        near_zero = model(**batch).logits
        model.gate.proj.bias.fill_(40.0)  # g ~ c: every visual key suppressed
        suppressed = model(**batch).logits
    assert not torch.allclose(near_zero, suppressed)  # softmax_1 is not bias-invariant
    base = make_model("base").eval()
    assert base.gate is None and not base.deletion.use_softmax_1 and model.deletion.use_softmax_1


@pytest.mark.parametrize(
    "method,rate", [("dtm", DELTA), ("fastv", None), ("phi_r", None), ("random", None), ("base", None)]
)
def test_cache_shrinks_by_budget(method, rate):
    """Layers 1..5 cache the full prompt, layers 6.. only the compacted one (Sec. 4.2)."""
    model = make_model(method).eval()
    batch = prompt()
    n = batch["input_ids"].shape[1]
    _, cache, _ = model.prefill(**batch, deletion_rate=rate)
    removed = 0 if method == "base" else math.floor(DELTA * N_IMAGE)
    for i in range(tiny_config().text_config.num_hidden_layers):
        assert cache.seq_length(i) == (n if i < SPLIT else n - removed)
    kept = cache.regions[1]
    assert int(kept.visual.sum(1)[0]) == N_IMAGE - removed
    assert torch.all(kept.pos.diff(dim=1) > 0)  # survivors keep original positions


def _uncached_greedy(model, batch: dict, steps: int, rate) -> torch.Tensor:
    """Greedy decoding by re-running the whole sequence without a cache."""
    ids = batch["input_ids"]
    labels = torch.full_like(ids, -100)
    for _ in range(steps):
        out = model(
            input_ids=ids, image_embeds=batch["image_embeds"], labels=labels,
            deletion_rate=rate, hard_delete=True,
        )
        last = -1 if out.kept_index is None else (out.kept_index >= 0).sum(1) - 1  # rows may keep different counts
        rows = torch.arange(ids.shape[0])
        nxt = out.logits[rows, last].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        labels = torch.cat([labels, nxt], dim=1)  # generated tokens are answer queries
    return ids


@pytest.mark.parametrize(
    "method,rate", [("dtm", DELTA), ("dtm", None), ("fastv", None), ("phi_r", None), ("base", None)]
)
def test_cached_generate_matches_uncached(method, rate):
    """generate() (prefill deletion + cached decoding) == uncached greedy recomputation."""
    model = make_model(method).eval()
    batch = prompt()
    if method == "dtm":
        model.calibrate_gate(**batch, delta_init=0.5)
    steps = 6
    cached = model.generate(**batch, deletion_rate=rate, hard_delete=True, max_new_tokens=steps, eos_token_id=[])
    assert torch.equal(cached, _uncached_greedy(model, batch, steps, rate))


def test_generate_left_padding():
    """A left-padded row decodes exactly as it does alone (masks and positions ignore padding)."""
    model = make_model("dtm").eval()
    long, short = prompt(batch=1, question=6, seed=3), prompt(batch=1, question=3, seed=4)
    model.calibrate_gate(**long, delta_init=0.5)
    pad = long["input_ids"].shape[1] - short["input_ids"].shape[1]
    ids = torch.cat([long["input_ids"], torch.cat([torch.zeros(1, pad, dtype=torch.long), short["input_ids"]], 1)])
    mask = (torch.arange(ids.shape[1]) >= torch.tensor([[0], [pad]])).long()
    embeds = torch.cat([long["image_embeds"], short["image_embeds"]])
    both = model.generate(input_ids=ids, attention_mask=mask, image_embeds=embeds, deletion_rate=DELTA,
                          max_new_tokens=5, eos_token_id=[])
    alone = model.generate(**short, deletion_rate=DELTA, max_new_tokens=5, eos_token_id=[])
    assert torch.equal(both[1, -5:], alone[0, -5:])


@pytest.mark.parametrize("method", ["fastv", "phi_r", "random"])
@pytest.mark.parametrize("aggregation", ["all_mean", "text_mean", "text_last"])
def test_baselines_delete_exact_budget(method, aggregation):
    """Baselines delete exactly floor(delta n_v) visual tokens, in training and at inference (Sec. 5)."""
    if method == "random" and aggregation != "all_mean":
        pytest.skip("aggregation does not apply to Random")
    model = make_model(method, aggregation=aggregation)
    batch = with_answer(prompt())
    ids = batch["input_ids"]
    for training, rate in ((True, None), (False, 0.25), (False, 0.9)):
        model.train(training)
        out = model(**batch, deletion_rate=rate, hard_delete=not training)
        removed = math.floor((DELTA if rate is None else rate) * N_IMAGE)
        assert out.gate_scores is None and out.kept_index.shape == (2, ids.shape[1] - removed)
        kept_ids = torch.gather(ids, 1, out.kept_index)
        assert torch.all((kept_ids == IMG).sum(1) == N_IMAGE - removed)
        assert torch.all((kept_ids != IMG).sum(1) == (ids != IMG).sum(1))  # text, <boi>, <eoi> never deleted
        if training:
            assert torch.isfinite(out.loss)
            out.loss.backward()


def test_dtm_natural_threshold():
    """DTM without a rate deletes exactly the visual tokens with g <= tau (Sec. 3.2, App. A.5)."""
    model = make_model("dtm").eval()
    batch = prompt()
    model.calibrate_gate(**batch, delta_init=0.5)
    with torch.no_grad():
        soft = model(**batch)
        hard = model(**batch, hard_delete=True)
    below = (soft.gate_scores <= -7.0) & soft.visual_mask
    kept = hard.kept_index
    for row in range(2):
        expected = torch.nonzero(~below[row]).squeeze(1)
        got = kept[row][kept[row] >= 0]
        assert torch.equal(got, expected)


def test_pixel_values_path():
    """Images go through the frozen SigLIP tower and projector into the image-token slots."""
    model = make_model("fastv").eval()
    batch = prompt()
    pixels = torch.randn(2, 3, 32, 32)
    out = model(input_ids=batch["input_ids"], pixel_values=pixels)
    assert out.logits.shape[1] == batch["input_ids"].shape[1] - math.floor(DELTA * N_IMAGE)
    feats = model.model.get_image_features(pixels, return_dict=True).pooler_output
    out2 = model(input_ids=batch["input_ids"], image_embeds=feats)
    assert torch.allclose(out.logits, out2.logits)


def test_base_matches_backbone():
    """Base reproduces stock Gemma 3 (causal, local-window and within-image bidirectional masks)."""
    from transformers import Gemma3ForConditionalGeneration

    model = make_model("base").eval()
    reference = Gemma3ForConditionalGeneration(model.config).eval()
    reference.load_state_dict(model.state_dict())
    reference.config.text_config._attn_implementation = "eager"
    batch = with_answer(prompt(question=12))  # longer than the local window
    ids, is_image = batch["input_ids"], batch["input_ids"] == IMG
    with torch.no_grad():
        embeds = reference.get_input_embeddings()(ids)
        embeds = embeds.masked_scatter(is_image[..., None].expand_as(embeds), batch["image_embeds"])
        expected = reference(inputs_embeds=embeds, token_type_ids=is_image.long(),
                             attention_mask=torch.ones_like(ids)).logits
        got = model(input_ids=ids, image_embeds=batch["image_embeds"]).logits
    assert torch.allclose(got, expected, atol=1e-5)
