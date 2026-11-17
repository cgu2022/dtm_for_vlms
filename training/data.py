"""Data pipeline for task fine-tuning and evaluation (Sec. 5; App. B.2 "Vision processing"; App. C).

Every example is one image, one question and its reference answers. Prompts are
zero-shot and place the image block before the question [Sec. 4.2]::

    <start_of_image>Question: {question}\nAnswer:

The backbone's own processor resizes each image to 896x896 and expands
``<start_of_image>`` into the BOI marker, 256 visual tokens and the EOI marker
[App. B.2]. The frozen SigLIP tower and projector inside the model embed the
pixels. Training targets are the first reference answer followed by EOS, at
most 32 tokens including EOS.

* T5Gemma 2 (encoder--decoder): the prompt is the encoder input and the target
  is the decoder label sequence.
* Gemma 3 (decoder-only): one sequence ``prompt target``; prompt positions get
  label ``-100``, and ``token_type_ids`` mark the image tokens so they attend to
  each other bidirectionally.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from evaluation.metrics import REC_PROMPT, format_box
from evaluation.tasks import CHECKPOINT_TASK, MAX_NEW_TOKENS, TASKS, TaskSpec, subsample_indices

IGNORE_INDEX = -100
MAX_TARGET_TOKENS = MAX_NEW_TOKENS  # answer + EOS [App. C]

Family = Literal["t5gemma2", "gemma3"]


@dataclass(frozen=True)
class BackboneSpec:
    """A pretrained backbone and its global training batch size [App. C]."""

    hf_id: str
    family: Family
    batch_size: int


BACKBONES: dict[str, BackboneSpec] = {
    "t5gemma2-1b-1b": BackboneSpec("google/t5gemma-2-1b-1b", "t5gemma2", 256),
    "t5gemma2-270m-270m": BackboneSpec("google/t5gemma-2-270m-270m", "t5gemma2", 256),
    "gemma3-4b": BackboneSpec("google/gemma-3-4b-pt", "gemma3", 128),
}


@dataclass
class VQAExample:
    image: Image.Image
    question: str  # fully formatted prompt text (without the image block)
    answers: list[str]  # references; answers[0] is the training target


# --------------------------------------------------------------------------- #
# Dataset adapter over evaluation/tasks.py
# --------------------------------------------------------------------------- #
#: The Cauldron (training source for InfoVQA, VQAv2 and TallyQA) appends one of
#: these terse-answer instructions to every question and ends every answer with
#: a period; both are removed so training examples match the evaluation format.
_CAULDRON_SUFFIXES = (
    "Concise answer only.", "Give a very brief answer.", "Answer briefly.", "Short answer required.",
    "Quick response, please.", "Write a very short answer.", "Be succinct.", "Keep it brief.",
    "Keep it short and to the point.", "Make the answer very short.", "Answer the question with a short phrase.",
)


def clean_cauldron_question(text: Any) -> str:
    cleaned = str(text or "").strip()
    for suffix in _CAULDRON_SUFFIXES:
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)].rstrip()
    return cleaned


def clean_cauldron_answer(text: Any) -> str:
    cleaned = str(text or "").strip()
    return cleaned[:-1].rstrip() if cleaned.endswith(".") else cleaned


def _to_image(value: Any) -> Image.Image:
    if isinstance(value, list):
        value = value[0]
    if isinstance(value, dict):
        value = Image.open(io.BytesIO(value["bytes"])) if value.get("bytes") else Image.open(value["path"])
    elif isinstance(value, str):
        value = Image.open(value)
    if not isinstance(value, Image.Image):
        raise TypeError(f"unsupported image value of type {type(value).__name__}")
    return value.convert("RGB")


def _to_answers(value: Any) -> list[str]:
    if isinstance(value, (str, int, float)):
        return [str(value)]
    return [str(item["answer"]) if isinstance(item, dict) else str(item) for item in value]


def _expressions(value: Any) -> list[str]:
    """Referring expressions from a string, a ``{"sent"/"raw"}`` dict, or a list of either."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        text = str(value.get("sent") or value.get("raw") or "").strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [e for item in value for e in _expressions(item)]
    return []


def _box_answer(bbox: Sequence[float], fmt: str, image: Image.Image) -> str:
    """Pixel box (``"xywh"`` or ``"xyxy"``) -> normalised ``[x1, y1, x2, y2]`` target string."""
    x1, y1, a, b = (float(v) for v in bbox[:4])
    x2, y2 = (x1 + a, y1 + b) if fmt == "xywh" else (a, b)
    w, h = image.size
    return format_box([min(max(v, 0.0), 1.0) for v in (x1 / w, y1 / h, x2 / w, y2 / h)])


@dataclass(frozen=True)
class _Record:
    row: int  # row of the Hub split holding the image
    question: str
    answers: tuple[str, ...] = ()
    box: Optional[tuple[float, ...]] = None  # RefCOCO+: pixel box of the referent
    box_format: str = "xywh"


class TaskDataset(Dataset):
    """Image-question-answers view of a task split from the Hugging Face Hub.

    ``split="eval"`` loads the task's evaluation split (``TaskSpec.eval_*``) with
    its seeded subsample (``subsample_indices``); ``split="train"`` loads the
    fine-tuning split. ``limit`` keeps a seeded uniform subsample. Sources:

    * multi-turn Cauldron rows (``texts``): one example per question-answer turn
      (InfoVQA: the first turn of each row), instructions and final periods removed;
    * TallyQA test (``qa``): one example per question about an image;
    * GQA: questions joined to the ``*_all_images`` config by ``imageId``;
    * RefCOCO+: one example per referring expression, target = normalised box;
      the training export references COCO images by id, which are read from
      ``coco_image_dir`` (default: the ``COCO_IMAGE_DIR`` environment variable,
      a folder of COCO images named ``<12-digit id>.jpg`` or
      ``COCO_train2014_<12-digit id>.jpg``);
    * all other tasks: one row per example with a question and its answers.
    """

    def __init__(
        self,
        task: str,
        split: Literal["train", "eval"],
        limit: Optional[int] = None,
        coco_image_dir: Optional[str] = None,
    ) -> None:
        import datasets

        self.task = task
        self.spec: TaskSpec = TASKS[task]
        if split == "train":
            source = TASKS[CHECKPOINT_TASK.get(task, task)]
            if source.train_dataset is None:
                raise ValueError(f"task {task!r} has no training split")
            dataset, config, hf_split = source.train_dataset, source.train_config, "train"
            cap = limit
        else:
            dataset, config, hf_split = self.spec.eval_dataset, self.spec.eval_config, self.spec.eval_split
            cap = min(limit, self.spec.num_eval_examples) if limit else self.spec.num_eval_examples
        self.rows = datasets.load_dataset(dataset, config, split=hf_split)
        self.coco_image_dir = coco_image_dir or os.environ.get("COCO_IMAGE_DIR")
        self._gqa_images = None
        if "imageId" in self.rows.column_names:  # GQA keeps images in a companion config
            table = datasets.load_dataset(dataset, config.replace("balanced_instructions", "all_images"), split=hf_split)
            ids = table.select_columns(["id"]).to_dict()["id"]
            self._gqa_images = (table, {str(k): i for i, k in enumerate(ids)})
        self.records = self._records()
        keep = subsample_indices(len(self.records), cap)
        if keep is not None:
            self.records = [self.records[i] for i in keep]

    def _column(self, name: str) -> list:
        return self.rows.select_columns([name]).to_dict()[name]

    def _records(self) -> list[_Record]:
        cols = set(self.rows.column_names)
        out: list[_Record] = []
        if "texts" in cols:  # The Cauldron
            for i, turns in enumerate(self._column("texts")):
                for turn in (turns or [])[: 1 if self.task == "infovqa" else None]:
                    q, a = clean_cauldron_question(turn.get("user")), clean_cauldron_answer(turn.get("assistant"))
                    if q and a:
                        out.append(_Record(i, q, (a,)))
        elif "qa" in cols:  # TallyQA test
            for i, items in enumerate(self._column("qa")):
                for item in items or []:
                    q, a = str(item.get("question") or "").strip(), item.get("answer")
                    if q and a is not None:
                        out.append(_Record(i, q, (str(a).strip(),)))
        elif self.spec.metric == "acc_iou50":  # RefCOCO+
            train = "sentences" in cols
            texts = self._column("sentences" if train else "answer")
            for i, (bbox, value) in enumerate(zip(self._column("bbox"), texts)):
                if bbox and len(bbox) >= 4:
                    for expression in _expressions(value):
                        out.append(_Record(i, expression, box=tuple(bbox[:4]), box_format="xyxy" if train else "xywh"))
        else:
            q_field = next(f for f in ("question", "query") if f in cols)
            a_field = next(f for f in ("answers", "answer", "label") if f in cols)
            for i, (q, a) in enumerate(zip(self._column(q_field), self._column(a_field))):
                out.append(_Record(i, str(q), tuple(_to_answers(a))))
        return out

    def _image(self, row: int) -> Image.Image:
        if self._gqa_images is not None:
            table, ids = self._gqa_images
            return _to_image(table[ids[str(self.rows[row]["imageId"])]]["image"])
        item = self.rows[row]
        if "image" in item or "images" in item:
            return _to_image(item.get("image", item.get("images")))
        return _to_image(self._coco_path(item))

    def _coco_path(self, item: dict) -> str:
        if not self.coco_image_dir:
            raise ValueError("RefCOCO+ training images come from COCO: set COCO_IMAGE_DIR or pass coco_image_dir")
        image_id = int(item.get("image_id") or str(item["file_name"]).rsplit("_", 1)[-1].split(".")[0])
        for name in (f"{image_id:012d}.jpg", f"COCO_train2014_{image_id:012d}.jpg"):
            path = os.path.join(self.coco_image_dir, name)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"COCO image {image_id} not found in {self.coco_image_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> VQAExample:
        rec = self.records[i]
        image = self._image(rec.row)
        if rec.box is not None:  # referring expression -> normalised box [App. C]
            return VQAExample(image, REC_PROMPT.format(expression=rec.question), [_box_answer(rec.box, rec.box_format, image)])
        return VQAExample(image, self.spec.prompt.format(question=rec.question), list(rec.answers))


# --------------------------------------------------------------------------- #
# Collators
# --------------------------------------------------------------------------- #
class VQACollator:
    """Turns :class:`VQAExample` lists into model inputs for one backbone family.

    ``train=True`` adds target labels (answer + EOS, at most 32 tokens);
    ``train=False`` returns prompt-only batches for ``generate`` (left-padded
    for the decoder-only Gemma 3).
    """

    def __init__(self, processor: Any, family: Family, train: bool, max_target_tokens: int = MAX_TARGET_TOKENS) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.family = family
        self.train = train
        self.max_target_tokens = max_target_tokens

    def _target_ids(self, answer: str) -> list[int]:
        # Gemma 3 continues the prompt "...Answer:", so the target starts with a space.
        text = answer.strip() if self.family == "t5gemma2" else " " + answer.strip()
        ids = self.tokenizer(text, add_special_tokens=False).input_ids[: self.max_target_tokens - 1]
        return ids + [self.tokenizer.eos_token_id]

    def _encode_prompts(self, examples: Sequence[VQAExample], padding_side: str) -> dict[str, torch.Tensor]:
        self.tokenizer.padding_side = padding_side
        texts = [self.processor.boi_token + e.question for e in examples]  # image block, then question [Sec. 4.2]
        enc = dict(self.processor(text=texts, images=[[e.image] for e in examples], padding=True, return_tensors="pt"))
        if self.family == "t5gemma2":
            enc.pop("token_type_ids", None)  # the encoder is bidirectional
        return enc

    def __call__(self, examples: Sequence[VQAExample]) -> dict[str, torch.Tensor]:
        if not self.train:
            return self._encode_prompts(examples, "left" if self.family == "gemma3" else "right")
        enc = self._encode_prompts(examples, "right")
        targets = [self._target_ids(e.answers[0]) for e in examples]
        if self.family == "t5gemma2":
            width = max(len(t) for t in targets)
            enc["labels"] = torch.tensor([t + [IGNORE_INDEX] * (width - len(t)) for t in targets])
            return enc
        return self._append_targets(enc, targets)

    def _append_targets(self, enc: dict[str, torch.Tensor], targets: list[list[int]]) -> dict[str, torch.Tensor]:
        """Gemma 3: ``[bos] prompt target [pad...]``, loss on the target only."""
        pad = self.tokenizer.pad_token_id
        rows = []
        for i, target in enumerate(targets):
            n = int(enc["attention_mask"][i].sum())
            rows.append((enc["input_ids"][i, :n].tolist(), enc["token_type_ids"][i, :n].tolist(), target))
        width = max(len(p) + len(t) for p, _, t in rows)
        ids, types, labels, mask = [], [], [], []
        for prompt, tt, target in rows:
            fill = width - len(prompt) - len(target)
            ids.append(prompt + target + [pad] * fill)
            types.append(tt + [0] * (len(target) + fill))
            labels.append([IGNORE_INDEX] * len(prompt) + target + [IGNORE_INDEX] * fill)
            mask.append([1] * (len(prompt) + len(target)) + [0] * fill)
        enc.update(
            input_ids=torch.tensor(ids),
            token_type_ids=torch.tensor(types),
            labels=torch.tensor(labels),
            attention_mask=torch.tensor(mask),
        )
        return enc


def load_processor(backbone: str) -> Any:
    """The backbone's own processor (896x896 images, 256 visual tokens) [App. B.2]."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(BACKBONES[backbone].hf_id)


__all__ = ["BACKBONES", "BackboneSpec", "IGNORE_INDEX", "MAX_TARGET_TOKENS", "TaskDataset", "VQACollator",
           "VQAExample", "load_processor"]
