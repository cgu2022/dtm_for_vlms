"""Inference protocol and the two evaluation settings (Sec. 5 "Two evaluation settings"; App. C).

* Matched rate: each checkpoint is evaluated at its own training rate ``delta``.
* Fixed checkpoint: one checkpoint is evaluated at 10%, 20%, ..., 90% deletion.

At inference every method deletes exactly ``floor(delta * n_v)`` of the ``n_v``
visual tokens of each example (hard deletion, exact budget); for DTM this
separates the budget from the gate's natural threshold crossing [Sec. 5,
App. A.5]. ``--natural`` instead runs DTM in its natural mode (delete
``g_j <= tau``). Decoding is greedy with at most 32 new tokens; deletion is
applied once, on the encoder input (T5Gemma 2) or the prompt prefill (Gemma 3).

FLOPs savings [Sec. 5 "Computational cost", App. G] come from the recorded per-example lengths:
non-visual input tokens, visual tokens, deleted visual tokens and generated
tokens (including the stop token).

Usage::

    python -m evaluation.evaluate --checkpoint RUN/best --setting matched
    python -m evaluation.evaluate --checkpoint RUN/best --setting fixed --output fixed.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation.flops import BACKBONES as FLOPS_BACKBONES
from evaluation.flops import flops_saved
from evaluation.metrics import corpus_score
from evaluation.tasks import MAX_NEW_TOKENS, TASKS
from training.data import BACKBONES, Family, TaskDataset, VQACollator, load_processor
from training.trainer import load_checkpoint

FIXED_CHECKPOINT_RATES = tuple(round(0.1 * i, 1) for i in range(1, 10))


@dataclass
class Prediction:
    prediction: str
    references: list[str]
    n_text: int  # unpadded non-visual input tokens (prompt, BOS, image boundary markers)
    n_visual: int
    n_deleted: Optional[int]
    n_generated: int  # generated tokens including the stop token


def _generated_length(tokens: Sequence[int], eos_ids: set[int]) -> int:
    for i, t in enumerate(tokens):
        if t in eos_ids:
            return i + 1
    return len(tokens)


@torch.no_grad()
def predict(
    model: nn.Module,
    processor: Any,
    family: Family,
    dataset: Dataset,
    deletion_rate: Optional[float],
    batch_size: int = 8,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[Prediction]:
    """Greedy decoding with hard deletion; ``deletion_rate=None`` is DTM's natural mode (or Base)."""
    collate = VQACollator(processor, family, train=False)
    loader = DataLoader(range(len(dataset)), batch_size=batch_size, collate_fn=lambda idx: [dataset[i] for i in idx])
    device = next(model.parameters()).device
    tok = processor.tokenizer
    stop = model.generation_config.eos_token_id
    eos_ids = {tok.eos_token_id, tok.pad_token_id, *(stop if isinstance(stop, list) else [stop])} - {None}
    deletes = model.deletion.deletes
    rate = deletion_rate if deletes else None
    out: list[Prediction] = []
    for examples in loader:
        batch = {k: v.to(device) for k, v in collate(examples).items()}
        generated = model.generate(
            **batch, deletion_rate=rate, hard_delete=True, max_new_tokens=max_new_tokens, do_sample=False
        )
        # Gemma 3 returns prompt + answer; T5Gemma 2 returns the decoder start token + answer.
        start = batch["input_ids"].shape[1] if family == "gemma3" else 1
        generated = generated[:, start:].tolist()
        n_visual = (batch["input_ids"] == processor.image_token_id).sum(1).tolist()
        n_input = batch["attention_mask"].sum(1).tolist()
        for ex, gen, nv, ni in zip(examples, generated, n_visual, n_input):
            m = _generated_length(gen, eos_ids)
            out.append(Prediction(
                prediction=tok.decode(gen[:m], skip_special_tokens=True).strip(),
                references=ex.answers,
                n_text=int(ni - nv),
                n_visual=int(nv),
                n_deleted=(math.floor(rate * nv) if rate is not None else (0 if not deletes else None)),  # [Sec. 5]
                n_generated=m,
            ))
    return out


def summarize(predictions: list[Prediction], metric: str, backbone: str, method: str, k: int) -> dict[str, Any]:
    """Task score (0-100) and FLOPs saved relative to the unpruned backbone."""
    result: dict[str, Any] = {
        "score": corpus_score(metric, [p.prediction for p in predictions], [p.references for p in predictions]),
        "num_examples": len(predictions),
    }
    if all(p.n_deleted is not None for p in predictions):
        lengths = [(p.n_text, p.n_visual, p.n_deleted, p.n_generated) for p in predictions]
        result["flops_saved"] = flops_saved(FLOPS_BACKBONES[backbone], method, lengths, k=k)
    return result


def evaluate_model(
    model: nn.Module,
    processor: Any,
    backbone: str,
    dataset: TaskDataset,
    rates: Sequence[Optional[float]],
    batch_size: int = 8,
) -> dict[str, dict[str, Any]]:
    """Scores ``model`` on ``dataset`` at each deletion rate (``None`` = natural mode / no deletion)."""
    deletion = model.deletion
    results = {}
    for rate in rates:
        preds = predict(model, processor, BACKBONES[backbone].family, dataset, rate, batch_size=batch_size)
        summary = summarize(preds, dataset.spec.metric, backbone, deletion.method, deletion.delete_after_layer)
        summary["predictions"] = [asdict(p) for p in preds]
        results["natural" if rate is None else f"{rate:.1f}"] = summary
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="directory written by training.train (e.g. RUN/best)")
    parser.add_argument("--setting", choices=("matched", "fixed"), default="matched")
    parser.add_argument("--rates", type=float, nargs="*", help="override the setting's deletion rates")
    parser.add_argument("--natural", action="store_true", help="DTM only: delete g <= tau instead of an exact budget")
    parser.add_argument("--task", help="evaluation task (default: the training task; 'pope' for VQAv2 checkpoints)")
    parser.add_argument("--limit", type=int, help="evaluate a seeded subsample of this many examples")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--output", default=None, help="JSON results file (default: CHECKPOINT/eval_<setting>.json)")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(args.checkpoint, dtype=getattr(torch, args.dtype), device=device)
    task = args.task or meta["task"]
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}")
    delta = float(meta["deletion"]["delta"])
    if model.deletion.method == "base":
        rates: list[Optional[float]] = [None]
    elif args.natural:
        if model.deletion.method != "dtm":
            raise SystemExit("--natural applies to DTM only")
        rates = [None]
    elif args.rates:
        rates = list(args.rates)
    else:
        rates = [delta] if args.setting == "matched" else list(FIXED_CHECKPOINT_RATES)

    processor = load_processor(meta["backbone"])
    dataset = TaskDataset(task, "eval", limit=args.limit)
    results = evaluate_model(model, processor, meta["backbone"], dataset, rates, batch_size=args.batch_size)

    output = Path(args.output) if args.output else Path(args.checkpoint) / f"eval_{args.setting}_{task}.json"
    output.write_text(json.dumps({"task": task, "setting": args.setting, **meta, "results": results}, indent=1))
    for key, r in results.items():
        flops = f"  FLOPs saved {r['flops_saved']:.1f}%" if "flops_saved" in r else ""
        print(f"{task} rate={key}: score {r['score']:.2f} (n={r['num_examples']}){flops}")


if __name__ == "__main__":
    main()
