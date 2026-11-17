"""Task fine-tuning entry point (Sec. 5; App. C).

Builds the backbone with its deletion method via ``from_pretrained_dtm``, adds
LoRA to the language model (the vision tower stays frozen), trains with
:class:`training.trainer.DTMTrainer` and writes ``OUTPUT_DIR/best`` (best
validation score at the trained rate) and ``OUTPUT_DIR/final``, each holding
the LoRA adapter, the gate weights (DTM) and the deletion settings.

Example::

    python -m training.train --backbone t5gemma2-1b-1b --method dtm --task docvqa \\
        --delta 0.6 --max_steps 300 --output_dir runs/docvqa_dtm_0.6
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import torch
from torch import nn
from transformers import TrainingArguments, set_seed

from dtm.config import AGGREGATIONS, METHODS, DeletionConfig
from evaluation.evaluate import predict, summarize
from training.data import BACKBONES, TaskDataset, VQACollator, load_processor
from training.trainer import DTMTrainer, HParams, add_lora, dtm_model, load_backbone


def build_model(backbone: str, deletion: DeletionConfig, hp: HParams, dtype: torch.dtype) -> nn.Module:
    """Pretrained backbone + deletion method + LoRA on the language model [Sec. 5, App. C]."""
    return add_lora(load_backbone(backbone, deletion, dtype), hp)


def training_arguments(args: argparse.Namespace, hp: HParams, batch_size: int) -> TrainingArguments:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation = max(1, batch_size // (args.per_device_batch_size * world))
    return TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type,
        warmup_steps=hp.warmup_steps,
        weight_decay=hp.weight_decay,
        max_grad_norm=hp.max_grad_norm,
        bf16=hp.bf16 and torch.cuda.is_available(),
        eval_strategy="steps" if args.eval_every > 0 else "no",
        eval_steps=args.eval_every if args.eval_every > 0 else None,
        save_strategy="no",  # best/ and final/ are written by DTMTrainer
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=args.seed,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--delta", type=float, default=0.6, help="training deletion rate (target for DTM)")
    p.add_argument("--delta_init", type=float, default=None, help="DTM gate calibration rate (default: delta)")
    p.add_argument("--delete_after_layer", type=int, default=5, help="L5: the gate reads the output of layer 5")
    p.add_argument("--aggregation", choices=AGGREGATIONS, default="all_mean", help="FastV / phi-R query pooling")
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=None, help="global batch (default: 256; Gemma 3 4B: 128)")
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=50, help="validation cadence in steps (0 disables)")
    p.add_argument("--eval_limit", type=int, default=None, help="subsample the validation split")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    spec = BACKBONES[args.backbone]
    hp = HParams(batch_size=args.batch_size)
    deletion = DeletionConfig(
        method=args.method, delete_after_layer=args.delete_after_layer, delta=args.delta,
        delta_init=args.delta_init, aggregation=args.aggregation, seed=args.seed,
    )
    dtype = torch.bfloat16 if (hp.bf16 and torch.cuda.is_available()) else torch.float32
    model = build_model(args.backbone, deletion, hp, dtype)
    model.print_trainable_parameters()

    processor = load_processor(args.backbone)
    train_set = TaskDataset(args.task, "train", limit=args.train_limit)
    validation = TaskDataset(args.task, "eval", limit=args.eval_limit) if args.eval_every > 0 else None

    def eval_fn(m: nn.Module) -> float:
        """Validation score with exact-budget hard deletion at the trained rate [App. C]."""
        preds = predict(dtm_model(m), processor, spec.family, validation, deletion.delta,
                        batch_size=args.eval_batch_size)
        summary = summarize(preds, validation.spec.metric, args.backbone, deletion.method, deletion.delete_after_layer)
        return summary["score"]

    trainer = DTMTrainer(
        model=model,
        args=training_arguments(args, hp, hp.batch_size or spec.batch_size),
        train_dataset=train_set,
        eval_dataset=validation,  # scored by eval_fn through DTMTrainer.evaluate
        data_collator=VQACollator(processor, spec.family, train=True),
        deletion=deletion,
        hparams=hp,
        backbone=args.backbone,
        task=args.task,
        eval_fn=eval_fn if validation is not None else None,
    )
    trainer.train()
    if validation is not None and args.max_steps % args.eval_every:
        trainer.evaluate()  # the last step is always a selection candidate
    trainer.save_final()


if __name__ == "__main__":
    main()
