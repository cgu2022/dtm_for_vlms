"""The ten benchmarks (Sec. 5 "Models and tasks"; App. C, Table 8).

Every method and deletion rate is evaluated on the same examples. Validation
splits are used wherever one exists; TallyQA and POPE have no validation split,
so their test splits are used. POPE evaluates the checkpoints fine-tuned on
VQAv2. VQAv2 and GQA are evaluated on fixed, seeded uniform subsamples of their
validation splits (:func:`subsample_indices`).

All prompts are zero-shot: the image block followed by ``PROMPT`` (or the
grounding prompt for RefCOCO+); decoding is greedy with at most 32 new tokens.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

PROMPT = "Question: {question}\nAnswer:"
MAX_NEW_TOKENS = 32
SUBSAMPLE_SEED = 20260825


@dataclass(frozen=True)
class TaskSpec:
    name: str
    metric: str  # key of evaluation.metrics.METRICS
    eval_dataset: str  # public dataset identifier (Hugging Face Hub)
    eval_config: Optional[str]
    eval_split: str
    num_eval_examples: int  # App. C
    train_dataset: Optional[str]
    train_config: Optional[str] = None
    evidence_regions: Optional[str] = None  # source used in Sec. 6.3 / App. E
    prompt: str = PROMPT


TASKS: Dict[str, TaskSpec] = {
    "docvqa": TaskSpec("docvqa", "anls", "lmms-lab/DocVQA", "DocVQA", "validation", 5349,
                       "pixparse/docvqa-single-page-questions", evidence_regions="OCR words"),
    "chartqa": TaskSpec("chartqa", "relaxed_accuracy", "HuggingFaceM4/ChartQA", None, "val", 1920,
                        "HuggingFaceM4/ChartQA", evidence_regions="support boxes (RefChartQA)"),
    "infovqa": TaskSpec("infovqa", "anls", "lmms-lab/DocVQA", "InfographicVQA", "validation", 2801,
                        "HuggingFaceM4/the_cauldron", "infographic_vqa", evidence_regions="OCR words"),
    "textvqa": TaskSpec("textvqa", "vqa_accuracy", "lmms-lab/textvqa", None, "validation", 5000,
                        "lmms-lab/textvqa", evidence_regions="OCR words (TextOCR)"),
    "vqav2": TaskSpec("vqav2", "vqa_accuracy", "lmms-lab/VQAv2", None, "validation", 25600,
                      "HuggingFaceM4/the_cauldron", "vqav2"),
    "vizwiz": TaskSpec("vizwiz", "vqa_accuracy", "lmms-lab/VizWiz-VQA", None, "val", 4319,
                       "Multimodal-Fatima/VizWiz_train", evidence_regions="answer-grounding polygons"),
    "gqa": TaskSpec("gqa", "gqa_accuracy", "lmms-lab/GQA", "val_balanced_instructions", "val", 12800,
                    "lmms-lab/GQA", "train_balanced_instructions", evidence_regions="scene-graph boxes"),
    "tallyqa": TaskSpec("tallyqa", "count_accuracy", "vikhyatk/tallyqa-test", None, "test", 38589,
                        "HuggingFaceM4/the_cauldron", "tallyqa"),
    "pope": TaskSpec("pope", "pope_accuracy", "lmms-lab/POPE", None, "test", 9000,
                     None, evidence_regions="COCO object boxes"),  # uses VQAv2-trained checkpoints
    "refcoco_plus": TaskSpec("refcoco_plus", "acc_iou50", "lmms-lab/RefCOCOplus", None, "val", 10758,
                             "jxu124/refcocoplus", evidence_regions="referent box"),
}

#: POPE has no training set of its own; it is scored with the VQAv2 checkpoints.
CHECKPOINT_TASK = {"pope": "vqav2"}


def subsample_indices(n_rows: int, limit: Optional[int], seed: int = SUBSAMPLE_SEED) -> Optional[List[int]]:
    """Seeded uniform subsample of ``limit`` rows, sorted back into split order.

    Returns None when the split already has at most ``limit`` rows.
    """
    if limit is None or limit <= 0 or n_rows <= 0 or limit >= n_rows:
        return None
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), int(limit)))


__all__ = ["TaskSpec", "TASKS", "CHECKPOINT_TASK", "PROMPT", "MAX_NEW_TOKENS", "subsample_indices"]
