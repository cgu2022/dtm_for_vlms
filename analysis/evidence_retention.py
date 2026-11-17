"""Spatial evidence retention (Sec. 6.3, Fig. 4; App. E.1).

For each question we mark the visual tokens that overlap a task-relevant
region ("evidence patches") and measure the fraction of them that a method
retains after deletion.

Source of the evidence regions
------------------------------
We did not create, draw, predict or edit any evidence region. Every box,
polygon and OCR word box comes unchanged from a publicly released dataset or
annotation set, as cited in Table 8 (App. C):

  ========  ============================================  ==========================================
  task      region geometry                               provenance
  ========  ============================================  ==========================================
  DocVQA    OCR word boxes                                released with MP-DocVQA (Tito et al., 2023)
  InfoVQA   OCR word boxes                                released with InfographicVQA (Mathew et al.)
  TextVQA   OCR word boxes                                TextOCR (Singh et al., 2021)
  ChartQA   support boxes                                 RefChartQA (Vogel et al., 2025)
  VizWiz    answer-grounding polygons                     VizWiz-VQA-Grounding (Chen et al., 2022)
  GQA       scene-graph object boxes                      GQA scene graphs (Hudson & Manning, 2019)
  POPE      object instance boxes                         COCO (Lin et al., 2015)
  RefCOCO+  referent box                                  RefCOCO+ (Yu et al., 2016)
  ========  ============================================  ==========================================

The DocVQA and InfoVQA OCR boxes are the OCR outputs distributed with those
datasets, and the RefChartQA support boxes were constructed by the RefChartQA
authors; we use all of them as released. VQAv2 and TallyQA have no evidence
regions and are not part of this analysis.

This module only decides *which* of the released regions belong to a
question, and maps them onto the visual-token grid:

  * DocVQA / InfoVQA / TextVQA: the released OCR words that spell a gold
    answer (:func:`match_answer_words`);
  * POPE: all instances of the queried object, for questions whose answer is
    "yes";
  * GQA: the scene-graph objects that the released GQA question annotations
    link to the answer, or otherwise to the question or its program;
  * ChartQA and RefCOCO+: the regions released for that question;
  * VizWiz: the released polygons for the question's most common answer.

Mapping regions to visual tokens
--------------------------------
Every image is resized to 896 x 896, and its 256 visual tokens tile the image
as a 16 x 16 grid in normalised coordinates (App. B.2), so a region in
normalised coordinates maps directly onto the grid. A patch counts as evidence
when the region covers at least a quarter of the patch, or when the patch holds
at least 30% of the region; the second clause keeps small regions such as a
single word on a large page. Polygons are rasterised in their native geometry
(:func:`mask_to_patches`), never replaced by enclosing rectangles.

Answer words are found by normalising OCR words and gold answers (lower case,
alphanumerics only), looking first for the answer as a run of consecutive OCR
words, then for its joined form, then for its informative tokens (numbers or
words of at least three letters outside a short stop list). Questions without
any evidence patch are not eligible.

Random deletion keeps each visual token with probability ``1 - delta``, so its
expected retention equals the kept fraction (dashed lines in Fig. 4).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

SIDE = 16
NUM_PATCHES = SIDE * SIDE
_STOP = {"the", "and", "for", "with", "from", "that", "this", "are", "was",
         "not", "you", "all", "any", "but", "has", "have", "its", "per", "who"}


def box_to_patches(x0: float, y0: float, x1: float, y1: float,
                   patch_frac: float = 0.25, box_frac: float = 0.30) -> np.ndarray:
    """``(256,)`` bool: grid patches covered by a normalised box (row-major)."""
    x0, x1 = sorted((float(np.clip(x0, 0, 1)), float(np.clip(x1, 0, 1))))
    y0, y1 = sorted((float(np.clip(y0, 0, 1)), float(np.clip(y1, 0, 1))))
    edges = np.arange(SIDE + 1) / SIDE
    ox = np.clip(np.minimum(x1, edges[1:]) - np.maximum(x0, edges[:-1]), 0, None)
    oy = np.clip(np.minimum(y1, edges[1:]) - np.maximum(y0, edges[:-1]), 0, None)
    inter = np.outer(oy, ox)
    box_area = max((x1 - x0) * (y1 - y0), 1e-12)
    m = (inter >= patch_frac / NUM_PATCHES) | (inter >= box_frac * box_area)
    return (m & (inter > 0)).ravel()


def mask_to_patches(mask: np.ndarray, patch_frac: float = 0.25, region_frac: float = 0.30) -> np.ndarray:
    """``(256,)`` bool: grid patches covered by a binary region mask (e.g. a rasterised polygon).

    The foreground fraction of each patch is integrated exactly over the pixel
    raster (also when the image size is not divisible by 16); the same
    quarter-of-patch / 30%-of-region rule as :func:`box_to_patches` applies.
    """
    m = np.asarray(mask, dtype=np.float64)
    h, w = m.shape
    integral = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys, xs = np.linspace(0, h, SIDE + 1), np.linspace(0, w, SIDE + 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h), np.minimum(x0 + 1, w)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    grid = ((1 - fy) * (1 - fx) * integral[np.ix_(y0, x0)] + fy * (1 - fx) * integral[np.ix_(y1, x0)]
            + (1 - fy) * fx * integral[np.ix_(y0, x1)] + fy * fx * integral[np.ix_(y1, x1)])
    coverage = np.clip(np.diff(np.diff(grid, axis=0), axis=1) / (w * h / NUM_PATCHES), 0, 1).ravel()
    return ((coverage >= patch_frac) | (coverage >= region_frac * coverage.sum())) & (coverage > 0)


def boxes_to_patches(boxes: Iterable[Sequence[float]]) -> np.ndarray:
    """Union of :func:`box_to_patches` over several regions."""
    m = np.zeros(NUM_PATCHES, bool)
    for b in boxes:
        m |= box_to_patches(*b[:4])
    return m


def normalise_word(w: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(w).lower())


def _informative(tok: str) -> bool:
    return bool(tok) and (any(ch.isdigit() for ch in tok) or (len(tok) >= 3 and tok not in _STOP))


def match_answer_words(words: Sequence[str], answers: Sequence[str]) -> list[int]:
    """Indices of the OCR words that spell one of the gold answers (empty if none)."""
    norm = [normalise_word(w) for w in words]
    fallback: list[int] = []
    for ans in answers:
        toks = [t for t in (normalise_word(x) for x in str(ans).split()) if t]
        if not toks:
            continue
        n = len(toks)
        for i in range(len(norm) - n + 1):  # consecutive run
            if norm[i:i + n] == toks:
                return list(range(i, i + n))
        joined = "".join(toks)  # split / joined OCR tokens
        if len(joined) >= 2:
            hit = [i for i, w in enumerate(norm) if w and w == joined]
            if hit:
                return hit
        if not fallback:  # informative tokens
            info = {t for t in toks if _informative(t)}
            fallback = [i for i, w in enumerate(norm) if w in info]
    return fallback


def ocr_evidence(words: Sequence[str], word_boxes: Sequence[Sequence[float]], answers: Sequence[str]) -> np.ndarray:
    """Evidence patches for OCR tasks: the boxes of the words that spell the answer."""
    idx = match_answer_words(words, answers)
    return boxes_to_patches(word_boxes[i] for i in idx)


def evidence_retention(keep: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    """Per-question fraction of evidence patches retained.

    Args:
        keep: ``(Q, 256)`` bool, visual tokens kept by a method (row-major grid).
        evidence: ``(Q, 256)`` bool evidence patches.

    Returns:
        ``(Q,)`` retention in [0, 1]; NaN for questions without evidence.
    """
    keep, evidence = np.asarray(keep, bool), np.asarray(evidence, bool)
    n_ev = evidence.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_ev > 0, (keep & evidence).sum(1) / n_ev, np.nan)


def mean_retention(keep: np.ndarray, evidence: np.ndarray) -> float:
    """Mean retention (percent) over eligible questions (those with evidence)."""
    r = evidence_retention(keep, evidence)
    return float(100.0 * np.nanmean(r))


def random_expectation(delta: float, n_visual: int = NUM_PATCHES) -> float:
    """Expected retention (percent) of Random deletion: the kept fraction."""
    n_deleted = int(np.floor(delta * n_visual))
    return 100.0 * (n_visual - n_deleted) / n_visual


__all__ = ["box_to_patches", "mask_to_patches", "boxes_to_patches", "match_answer_words", "ocr_evidence",
           "evidence_retention", "mean_retention", "random_expectation"]
