"""Smoke test of the training loop (Sec. 3.2, 4.3; App. C) on tiny random models, CPU, fp32.

For both backbone families and for DTM and FastV, three optimizer steps on
synthetic data check that:
* DTM calibrates the gate bias once, on the first training batch [Sec. 4.3];
* the PID controller updates alpha_t once per optimizer step [Eq. 3];
* the gate has its own optimizer parameter group [App. C];
* the loss stays finite, and a validation pass writes the best checkpoint.
"""

import math
from pathlib import Path

import pytest
import torch
from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig, TrainingArguments
from transformers.models.t5gemma2.configuration_t5gemma2 import (
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    T5Gemma2EncoderConfig,
)

from dtm.config import DeletionConfig
from dtm.models.gemma3 import VMrGemma3ForConditionalGeneration
from dtm.models.t5gemma2 import VMrT5Gemma2ForConditionalGeneration
from training.trainer import GATE_FILE, META_FILE, DTMTrainer, HParams, add_lora

PAD, EOS, BOS, BOI, EOI, IMG = 0, 1, 2, 97, 98, 99
STEPS, BATCH, DELTA = 3, 2, 0.6
BACKBONE = {"t5gemma2": "t5gemma2-270m-270m", "gemma3": "gemma3-4b"}  # names for checkpoint metadata / FLOPs


def _text_config(**extra) -> dict:
    return dict(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, query_pre_attn_scalar=16, sliding_window=8,
        pad_token_id=PAD, eos_token_id=EOS, bos_token_id=BOS, **extra,
    )


def _vision_config(image_size: int) -> dict:
    return dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                image_size=image_size, patch_size=4)


def tiny_t5gemma2(deletion: DeletionConfig) -> torch.nn.Module:
    """8+8 layers; 40x40 images -> 25 visual tokens."""
    encoder = T5Gemma2EncoderConfig(text_config=_text_config(), vision_config=_vision_config(40),
                                    mm_tokens_per_image=25, boi_token_index=BOI, eoi_token_index=EOI,
                                    image_token_index=IMG)
    config = T5Gemma2Config(encoder=encoder, decoder=T5Gemma2DecoderConfig(**_text_config()), image_token_index=IMG)
    return VMrT5Gemma2ForConditionalGeneration(config, deletion)


def tiny_gemma3(deletion: DeletionConfig) -> torch.nn.Module:
    """8 layers; 32x32 images -> 16 visual tokens."""
    config = Gemma3Config(text_config=Gemma3TextConfig(**_text_config(max_position_embeddings=256)),
                          vision_config=SiglipVisionConfig(**_vision_config(32)), mm_tokens_per_image=16,
                          boi_token_index=BOI, eoi_token_index=EOI, image_token_index=IMG)
    return VMrGemma3ForConditionalGeneration(config, deletion)


def synthetic_examples(family: str, n: int = STEPS * BATCH) -> list[dict[str, torch.Tensor]]:
    """``[bos] <boi> image <eoi> question`` prompts with 2-3 token answers + EOS."""
    g = torch.Generator().manual_seed(0)
    n_img, size = (25, 40) if family == "t5gemma2" else (16, 32)
    out = []
    for i in range(n):
        prompt = [BOS, BOI] + [IMG] * n_img + [EOI] + torch.randint(3, 90, (4 + i % 2,), generator=g).tolist()
        answer = torch.randint(3, 90, (2 + i % 2,), generator=g).tolist() + [EOS]
        ex = {"pixel_values": torch.randn(3, size, size, generator=g)}
        if family == "t5gemma2":
            ex.update(input_ids=torch.tensor(prompt), labels=torch.tensor(answer))
        else:
            ex.update(input_ids=torch.tensor(prompt + answer),
                      labels=torch.tensor([-100] * len(prompt) + answer),
                      token_type_ids=torch.tensor([int(t == IMG) for t in prompt] + [0] * len(answer)))
        out.append(ex)
    return out


def collate(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    width = {k: max(len(e[k]) for e in examples) for k in ("input_ids", "labels")}
    pad = lambda t, n, v: torch.cat([t, t.new_full((n - len(t),), v)])  # noqa: E731
    batch = {
        "input_ids": torch.stack([pad(e["input_ids"], width["input_ids"], PAD) for e in examples]),
        "attention_mask": torch.stack([pad(torch.ones_like(e["input_ids"]), width["input_ids"], 0) for e in examples]),
        "labels": torch.stack([pad(e["labels"], width["labels"], -100) for e in examples]),
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
    }
    if "token_type_ids" in examples[0]:
        batch["token_type_ids"] = torch.stack([pad(e["token_type_ids"], width["input_ids"], 0) for e in examples])
    return batch


@pytest.mark.parametrize("family", ["t5gemma2", "gemma3"])
@pytest.mark.parametrize("method", ["dtm", "fastv"])
def test_three_training_steps(family: str, method: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    deletion = DeletionConfig(method=method, delta=DELTA)
    base = (tiny_t5gemma2 if family == "t5gemma2" else tiny_gemma3)(deletion)
    hp = HParams(bf16=False)
    model = add_lora(base, hp)

    calibrations: list[float] = []
    if method == "dtm":
        calibrate = base.calibrate_gate
        base.calibrate_gate = lambda **kw: (calibrations.append(kw["delta_init"]), calibrate(**kw))[1]
        bias_before = base.gate.proj.bias.detach().clone()

    args = TrainingArguments(
        output_dir=str(tmp_path), max_steps=STEPS, per_device_train_batch_size=BATCH, learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type, weight_decay=hp.weight_decay, max_grad_norm=hp.max_grad_norm,
        logging_steps=1, save_strategy="no", eval_strategy="no", remove_unused_columns=False, report_to=[],
        use_cpu=True, dataloader_num_workers=0,
    )
    trainer = DTMTrainer(model=model, args=args, train_dataset=synthetic_examples(family), data_collator=collate,
                         deletion=deletion, hparams=hp, backbone=BACKBONE[family], task="docvqa",
                         eval_fn=lambda m: 50.0)
    alphas: list[float] = []
    if trainer.controller is not None:
        step = trainer.controller.step
        trainer.controller.step = lambda rate: (alphas.append(step(rate)), alphas[-1])[1]
    trainer.train()

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    assert len(losses) == STEPS and all(math.isfinite(x) for x in losses)

    groups = {g["name"]: g for g in trainer.optimizer.param_groups}
    gate_ids = {id(p) for g in groups.values() if g["name"].startswith("gate") for p in g["params"]}
    if method == "dtm":
        # [Sec. 4.3] one calibration, on the first batch, at delta_init = delta; bias moved.
        assert calibrations == [DELTA]
        assert not torch.equal(bias_before, base.gate.proj.bias.detach())
        # [Eq. 3] one controller update per optimizer step.
        assert trainer.controller.t == STEPS and len(alphas) == STEPS
        assert all(math.isfinite(a) and a >= 0 for a in alphas)
        # [App. C] gate parameters live only in their own groups, at the gate learning rate.
        assert gate_ids == {id(p) for p in base.gate.parameters()}
        assert all(g["initial_lr"] == hp.gate_learning_rate for g in groups.values() if g["name"].startswith("gate"))
        assert all(p.dtype == torch.float32 for p in base.gate.parameters())
    else:
        assert trainer.controller is None and base.gate is None and not gate_ids
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for g in groups.values() for p in g["params"]}
    assert not any("vision_tower" in n for n, p in model.named_parameters() if p.requires_grad)

    metrics = trainer.evaluate()
    assert metrics["eval_score"] == 50.0
    best = tmp_path / "best"
    assert (best / META_FILE).exists() and (best / "adapter_config.json").exists()
    assert (best / GATE_FILE).exists() == (method == "dtm")


class _StubTokenizer:
    eos_token_id, pad_token_id, padding_side = EOS, PAD, "right"

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids if i not in (EOS, PAD))


class _StubProcessor:
    """Stands in for the backbone processor: fixed-size synthetic prompts, left-padded for Gemma 3."""

    boi_token, image_token_id = "<boi>", IMG

    def __init__(self, family: str) -> None:
        self.family, self.tokenizer = family, _StubTokenizer()

    def __call__(self, text: list[str], images: list, padding: bool, return_tensors: str) -> dict[str, torch.Tensor]:
        examples = synthetic_examples(self.family, n=len(text))
        n_img = 25 if self.family == "t5gemma2" else 16
        prompts = [e["input_ids"][: n_img + 3 + 4 + i % 2] for i, e in enumerate(examples)]
        width = max(len(p) for p in prompts)
        left = self.tokenizer.padding_side == "left"
        ids = torch.full((len(prompts), width), PAD)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompts):
            sl = slice(width - len(p), width) if left else slice(0, len(p))
            ids[i, sl], mask[i, sl] = p, 1
        return {"input_ids": ids, "attention_mask": mask,
                "token_type_ids": (ids == IMG).long(),
                "pixel_values": torch.stack([e["pixel_values"] for e in examples])}


@pytest.mark.parametrize("family", ["t5gemma2", "gemma3"])
def test_predict_exact_budget(family: str) -> None:
    """Greedy decoding with exact-budget hard deletion; lengths feed the FLOPs accounting [Sec. 5]."""
    from PIL import Image

    from evaluation.evaluate import predict, summarize
    from training.data import VQAExample

    torch.manual_seed(0)
    model = (tiny_t5gemma2 if family == "t5gemma2" else tiny_gemma3)(DeletionConfig(method="dtm", delta=DELTA)).eval()
    data = [VQAExample(Image.new("RGB", (8, 8)), "q", ["1 2"]) for _ in range(3)]
    preds = predict(model, _StubProcessor(family), family, data, deletion_rate=0.4, batch_size=2, max_new_tokens=4)
    n_img = 25 if family == "t5gemma2" else 16
    assert len(preds) == 3
    assert all(p.n_visual == n_img and p.n_deleted == math.floor(0.4 * n_img) for p in preds)
    assert all(1 <= p.n_generated <= 4 and p.n_text == 3 + 4 + i % 2 for i, p in enumerate(preds))
    summary = summarize(preds, "anls", BACKBONE[family], "dtm", 5)
    assert 0.0 <= summary["score"] <= 100.0 and 0.0 < summary["flops_saved"] < 40.0
