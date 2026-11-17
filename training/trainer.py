"""Task fine-tuning for DTM and the baselines (Sec. 3.2, 4.1, 4.3, 5; App. C).

Objective [Sec. 3.2, 4.1]::

    L = L_CE + alpha_t * L_G,    L_G = mean of g_j over the batch's visual tokens,

with ``alpha_t`` set by the PID controller [Eq. 3] from the observed deletion rate
``r_t = #(g_j <= tau) / #visual`` of each optimizer step's batch (one update per
optimizer step). Before the first update, the gate bias is calibrated on the
first training batch so that a fraction ``delta_init`` of visual tokens starts
below ``tau`` [Sec. 4.3, App. A.2]. DTM trains with soft deletion throughout
[App. C]. FastV, phi-R and Random delete hard at the training rate ``delta``
(no gradient flows through their scores); Base deletes nothing [Sec. 5].

Optimisation [App. C, Table "Shared adaptation and gate recipe"]: AdamW with a
separate gate parameter group, one cosine schedule without warmup for adapters
and gate, weight decay only on weight matrices, gradient clipping at 1.

The trainer evaluates on the validation split at the trained rate (exact
budget, hard deletion) and keeps the checkpoint with the best score.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import nn
from transformers import Trainer, TrainerCallback

from dtm.config import DeletionConfig
from dtm.controller import PIDConfig, PIDController
from dtm.objective import gate_loss, observed_rate, total_loss

from .data import BACKBONES

GATE_FILE = "gate.pt"
META_FILE = "dtm_config.json"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class HParams:
    """App. C, Table "Shared adaptation and gate recipe"."""

    learning_rate: float = 1e-4  # adapters
    gate_learning_rate: float = 1e-4  # separate parameter group
    lr_scheduler_type: str = "cosine"  # adapters and gate share the schedule
    warmup_steps: int = 0
    weight_decay: float = 1e-4  # not applied to biases or norm scales
    batch_size: Optional[int] = None  # global; None -> 256 (Gemma 3 4B: 128)
    max_grad_norm: float = 1.0
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bf16: bool = True
    pid: PIDConfig = field(default_factory=PIDConfig)  # k_p=1, k_i=1e-5, k_d=0.5, omega=10, alpha_0=0.1


# --------------------------------------------------------------------------- #
# Model construction and checkpoints
# --------------------------------------------------------------------------- #
def load_backbone(backbone: str, deletion: DeletionConfig, dtype: torch.dtype) -> nn.Module:
    """Pretrained backbone with its deletion method, uniformly in ``dtype``; the gate stays fp32 [App. C]."""
    model = model_class(backbone).from_pretrained_dtm(BACKBONES[backbone].hf_id, deletion, dtype=dtype).to(dtype)
    if model.gate is not None:
        model.gate.float()
    return model


def model_class(backbone: str) -> type[nn.Module]:
    if BACKBONES[backbone].family == "t5gemma2":
        from dtm.models.t5gemma2 import VMrT5Gemma2ForConditionalGeneration

        return VMrT5Gemma2ForConditionalGeneration
    from dtm.models.gemma3 import VMrGemma3ForConditionalGeneration

    return VMrGemma3ForConditionalGeneration


def lora_target_regex() -> str:
    """Attention Q/K/V/O and MLP projections of the language model; the vision tower stays frozen [App. C]."""
    return r"^(?!.*vision_tower).*\.(" + "|".join(LORA_TARGETS) + r")$"


def add_lora(model: nn.Module, hp: HParams) -> nn.Module:
    """Wrap the language model with LoRA (r=32, alpha=64, dropout 0.05); the gate trains in full."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=hp.lora_r, lora_alpha=hp.lora_alpha, lora_dropout=hp.lora_dropout,
                        target_modules=lora_target_regex(), bias="none")
    peft_model = get_peft_model(model, config)
    if model.gate is not None:
        model.gate.float().requires_grad_(True)  # fp32 gate [App. C]
    return peft_model


def dtm_model(model: nn.Module) -> nn.Module:
    """The underlying DTM model of a (possibly LoRA-wrapped) model."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def save_checkpoint(model: nn.Module, output_dir: str | Path, backbone: str, task: str) -> None:
    """LoRA adapter, gate weights and the deletion settings."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    base = dtm_model(model)
    if base.gate is not None:
        torch.save({k: v.detach().cpu() for k, v in base.gate.state_dict().items()}, out / GATE_FILE)
    meta = {"backbone": backbone, "task": task, "deletion": asdict(base.deletion)}
    (out / META_FILE).write_text(json.dumps(meta, indent=2))


def load_checkpoint(
    checkpoint_dir: str | Path, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Base backbone + LoRA adapter (merged) + gate; returns the model and its metadata."""
    from peft import PeftModel

    ckpt = Path(checkpoint_dir)
    meta = json.loads((ckpt / META_FILE).read_text())
    deletion = DeletionConfig(**meta["deletion"])
    model = PeftModel.from_pretrained(load_backbone(meta["backbone"], deletion, dtype), ckpt).merge_and_unload()
    if model.gate is not None:
        model.gate.load_state_dict(torch.load(ckpt / GATE_FILE, map_location="cpu"))
    return model.to(device).eval(), meta


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class _StepEnd(TrainerCallback):
    """Runs the controller once per optimizer step."""

    def __init__(self, trainer: "DTMTrainer") -> None:
        self.trainer = trainer

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.trainer.controller_step()


class DTMTrainer(Trainer):
    """HF ``Trainer`` with the DTM objective, gate calibration, PID control and best-checkpoint selection.

    Args:
        deletion: the method and its settings (``deletion.delta`` is the training rate).
        hparams: App. C recipe.
        eval_fn: ``model -> score`` on the validation split at the trained rate
            (exact budget, hard deletion); ``None`` disables evaluation.
        backbone, task: recorded in saved checkpoints.
    """

    def __init__(
        self,
        *args: Any,
        deletion: DeletionConfig,
        hparams: HParams,
        backbone: str,
        task: str,
        eval_fn: Optional[Callable[[nn.Module], float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False  # CE is a token mean per micro-batch; Trainer averages over accumulation
        self.deletion = deletion
        self.hparams = hparams
        self.backbone = backbone
        self.task = task
        self.eval_fn = eval_fn
        self.controller: Optional[PIDController] = (
            PIDController(deletion.delta, hparams.pid) if deletion.method == "dtm" else None
        )
        self.calibrated = False
        self.best_score = -math.inf
        self._counts = torch.zeros(2, dtype=torch.float64)  # (deleted, visual) in the current optimizer step
        self._last: dict[str, float] = {}
        self.add_callback(_StepEnd(self))

    # ---------------------------------------------------------------- optimizer
    def create_optimizer(self, model: Optional[nn.Module] = None) -> torch.optim.Optimizer:
        """AdamW with adapter and gate groups; decay on weight matrices only [App. C]."""
        if self.optimizer is None:
            groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                is_gate = ".gate." in f".{name}"  # the delete gate, not the MLP's gate_proj
                decay = p.ndim >= 2  # no decay on biases and norm scales
                groups.setdefault((is_gate, decay), []).append(p)
            param_groups = [
                {
                    "params": params,
                    "lr": self.hparams.gate_learning_rate if is_gate else self.hparams.learning_rate,
                    "weight_decay": self.hparams.weight_decay if decay else 0.0,
                    "name": ("gate" if is_gate else "adapter") + ("" if decay else "_no_decay"),
                }
                for (is_gate, decay), params in sorted(groups.items())
            ]
            self.optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

    # --------------------------------------------------------------------- loss
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        method = self.deletion.method
        if method == "dtm" and not self.calibrated:
            if self.state.global_step == 0:
                with torch.no_grad():  # [Sec. 4.3, App. A.2] one-time bias calibration on the first batch
                    dtm_model(self.accelerator.unwrap_model(model)).calibrate_gate(
                        **inputs, delta_init=self.deletion.calibration_rate
                    )
            self.calibrated = True
        # Baselines delete hard at the training rate; DTM is soft; Base deletes nothing [Sec. 5].
        rate = self.deletion.delta if method in ("fastv", "phi_r", "random") else None
        outputs = model(**inputs, deletion_rate=rate, hard_delete=False)
        ce = outputs.loss
        loss = ce
        if method == "dtm":
            g, visual = outputs.gate_scores, outputs.visual_mask
            loss = total_loss(ce, g, visual, self.controller.alpha)  # [Sec. 3.2]
            n_visual = float(visual.sum())
            self._counts += torch.tensor([observed_rate(g, visual, self.deletion.tau) * n_visual, n_visual],
                                         dtype=torch.float64)
            self._last.update(gate_loss=float(gate_loss(g.detach(), visual)))
        self._last.update(ce_loss=float(ce.detach()))
        return (loss, outputs) if return_outputs else loss

    def controller_step(self) -> None:
        """One PID update per optimizer step from the step's pooled rate r_t [Eq. 3]."""
        if self.controller is None:
            return
        counts = self._counts.to(self.args.device)
        if self.args.world_size > 1:
            counts = self.accelerator.reduce(counts, reduction="sum")
        deleted, visual = counts.tolist()
        rate = deleted / max(visual, 1.0)
        self._last.update(deletion_rate=rate, alpha=self.controller.step(rate))
        self._counts.zero_()

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs:
            logs = {**logs, **self._last}
        super().log(logs, *args, **kwargs)

    # --------------------------------------------------------------- evaluation
    def evaluate(
        self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval"
    ) -> dict[str, float]:
        """Validation score at the trained rate; saves ``best/`` when it improves."""
        if self.eval_fn is None:
            return {}
        model = self.model
        was_training = model.training
        model.eval()
        with torch.no_grad():
            score = float(self.eval_fn(model))
        model.train(was_training)
        metrics = {f"{metric_key_prefix}_score": score}
        if self.is_world_process_zero() and score > self.best_score:
            self.best_score = score
            metrics[f"{metric_key_prefix}_best_step"] = float(self.state.global_step)
            save_checkpoint(model, Path(self.args.output_dir) / "best", self.backbone, self.task)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def save_final(self) -> None:
        if self.is_world_process_zero():
            save_checkpoint(self.model, Path(self.args.output_dir) / "final", self.backbone, self.task)


__all__ = ["DTMTrainer", "HParams", "LORA_TARGETS", "add_lora", "dtm_model", "load_backbone", "load_checkpoint",
           "model_class", "save_checkpoint"]
