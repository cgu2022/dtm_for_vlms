"""Tests for task metrics, FLOPs accounting and evidence retention."""

import math

import numpy as np
import pytest

from analysis.evidence_retention import (box_to_patches, evidence_retention, match_answer_words,
                                         random_expectation)
from evaluation import flops as F
from evaluation import metrics as M
from evaluation.tasks import TASKS, subsample_indices


def test_anls():
    assert M.anls("Hello  World", ["hello world"]) == 1.0
    assert M.anls("abcd", ["abce"]) == pytest.approx(0.75)
    assert M.anls("abcd", ["wxyz"]) == 0.0  # below the 0.5 cutoff


def test_relaxed_accuracy():
    assert M.relaxed_accuracy("104", ["100"]) == 1.0  # within 5%
    assert M.relaxed_accuracy("106", ["100"]) == 0.0
    assert M.relaxed_accuracy("50%", ["0.5"]) == 1.0
    assert M.relaxed_accuracy("Yes", ["yes"]) == 1.0
    assert M.relaxed_accuracy("1.26M", ["1.26"]) == 0.0


def test_vqa_accuracy_leave_one_out():
    refs = ["two"] * 3 + ["three"] * 7
    assert M.vqa_accuracy("2", refs) == pytest.approx(0.9)
    assert M.vqa_accuracy("3", refs) == 1.0
    assert M.gqa_accuracy("a cat", ["cat"]) == 1.0
    assert M.count_accuracy("There are 3.", ["3"]) == 1.0
    assert M.pope_accuracy("Yes, there is.", ["yes"]) == 1.0


def test_grounding_accuracy():
    gt = M.format_box([0.1, 0.1, 0.5, 0.5])
    assert M.grounding_accuracy("[0.1, 0.1, 0.5, 0.5]", [gt]) == 1.0
    assert M.grounding_accuracy("[0.6, 0.6, 0.9, 0.9]", [gt]) == 0.0
    assert M.iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == pytest.approx(0.5)


def test_task_table_matches_paper():
    expected = {"docvqa": 5349, "chartqa": 1920, "infovqa": 2801, "textvqa": 5000, "vqav2": 25600,
                "vizwiz": 4319, "gqa": 12800, "tallyqa": 38589, "pope": 9000, "refcoco_plus": 10758}
    assert {k: v.num_eval_examples for k, v in TASKS.items()} == expected
    idx = subsample_indices(1000, 100)
    assert len(idx) == 100 and idx == sorted(idx) and idx == subsample_indices(1000, 100)


def test_flops_formulas_by_hand():
    s = F.StackConfig(layers=12, d_model=8, d_ff=16, n_heads=2, n_kv=1, d_head=4, window=4)
    n = 10
    # layer 5 is local, layer 6 is global
    assert F.layer_flops(s, n, 5) == 4 * n * 8 * 4 * 3 + 4 * (n * 4) * 2 * 4 + 6 * n * 8 * 16
    assert F.layer_flops(s, n, 6) == 4 * n * 8 * 4 * 3 + 4 * (n * n) * 2 * 4 + 6 * n * 8 * 16


@pytest.mark.parametrize(
    "backbone, m, expected",
    [(F.T5GEMMA2_1B, 5.9, (14.9, 29.6, 44.3, 58.9)),
     (F.GEMMA3_4B, 5.8, (15.6, 31.1, 46.5, 62.0)),
     (F.T5GEMMA2_270M, 5.8, (13.9, 27.4, 40.6, 53.4))],
)
def test_flops_savings_docvqa(backbone, m, expected):
    """DocVQA mean lengths (18.9 text + 256 visual input tokens) reproduce the
    per-dataset FLOPs tables to within 0.2 points (the tables sum per example)."""
    for delta, ref in zip((0.2, 0.4, 0.6, 0.8), expected):
        saved = F.flops_saved(backbone, "dtm", [(18.9, 256, math.floor(delta * 256), m)])
        assert saved == pytest.approx(ref, abs=0.2)


def test_selection_costs():
    b = F.T5GEMMA2_1B
    assert F.selection_flops(b, "fastv", 275, 256, 5) == 0.0
    assert F.selection_flops(b, "dtm", 275, 256, 5) == 2 * 256 * 1152
    # layer 5 is local and N = 275 < W = 512, so P(N, 5) = N^2
    assert F.selection_flops(b, "phi_r", 275, 256, 5) == 2 * (275 * 275) * 4 * 256


def test_evidence_geometry_and_retention():
    m = box_to_patches(0.0, 0.0, 0.5, 0.5)
    assert m.reshape(16, 16)[:8, :8].all() and m.sum() == 64
    tiny = box_to_patches(0.30, 0.30, 0.305, 0.305)  # smaller than a patch: still one patch
    assert tiny.sum() == 1
    assert match_answer_words(["Total", "$1,234", "in", "2019"], ["1,234"]) == [1]
    keep = np.zeros((2, 256), bool)
    keep[:, :128] = True
    ev = np.zeros((2, 256), bool)
    ev[0, 120:136] = True
    r = evidence_retention(keep, ev)
    assert r[0] == pytest.approx(0.5) and math.isnan(r[1])
    assert random_expectation(0.8) == pytest.approx(100 * (256 - 204) / 256)
