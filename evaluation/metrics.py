"""Task metrics (App. C "Experimental Details").

Every metric is reported on a 0-100 scale (the functions below return per-example
values in [0, 1]; averages are multiplied by 100).

* ANLS (DocVQA, InfoVQA): average normalised Levenshtein similarity with the
  standard 0.5 cutoff.
* Relaxed accuracy (ChartQA): numeric answers are correct within 5% relative
  error; other answers require an exact (case-insensitive) match.
* VQA accuracy (TextVQA, VQAv2, VizWiz): the official VQA answer normalisation
  and the leave-one-annotator-out average of min(1, #matches / 3).
* GQA accuracy: exact match after the VQA answer normalisation.
* TallyQA exact-count accuracy: the first integer of the normalised answer.
* POPE accuracy: the first yes/no token of the answer.
* RefCOCO+ Acc@IoU>=0.5: a predicted box is correct when its intersection over
  union with the reference box is at least 0.5 (boxes as normalised
  ``[x1, y1, x2, y2]``).
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# ANLS
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def _anls_normalize(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(prediction: str, references: Sequence[str], threshold: float = 0.5) -> float:
    """Max NLS over the references, set to 0 below ``threshold``."""
    pred = _anls_normalize(prediction)
    best, any_ref = 0.0, False
    for ref in references:
        ref = _anls_normalize(ref)
        if not ref:
            continue
        any_ref = True
        denom = max(len(pred), len(ref))
        nls = 1.0 - levenshtein(pred, ref) / denom if denom else 1.0
        best = max(best, nls)
    if not any_ref:
        return 0.0
    return best if best >= threshold else 0.0


# --------------------------------------------------------------------------- #
# Relaxed accuracy (ChartQA)
# --------------------------------------------------------------------------- #
def _to_float(text: str) -> Optional[float]:
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def relaxed_accuracy(prediction: str, references: Sequence[str], max_relative_change: float = 0.05) -> float:
    """ChartQA relaxed accuracy: 5% relative tolerance for numbers, exact match otherwise."""
    pred = prediction.strip()
    for ref in references:
        ref = ref.strip()
        p, t = _to_float(pred), _to_float(ref)
        if p is not None and t:
            if abs(p - t) / abs(t) <= max_relative_change:
                return 1.0
        elif pred.lower() == ref.lower():
            return 1.0
    return 0.0


# --------------------------------------------------------------------------- #
# VQA accuracy and its normalisation (TextVQA, VQAv2, VizWiz; also GQA, TallyQA)
# --------------------------------------------------------------------------- #
_MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
               "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def _process_digit_article(text: str) -> str:
    words = []
    for w in text.lower().split():
        w = _MANUAL_MAP.get(w, w)
        if w not in _ARTICLES:
            words.append(w)
    return " ".join(_CONTRACTIONS.get(w, w) for w in words)


def normalize_vqa_answer(answer: str) -> str:
    """Official VQA answer normalisation (punctuation, number words, articles, contractions)."""
    text = (answer or "").replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(text))


def vqa_accuracy(prediction: str, references: Sequence[str]) -> float:
    """Official VQA accuracy: mean over annotators i of min(1, #{j != i : a_j == p} / 3)."""
    if not references:
        return 0.0
    pred = normalize_vqa_answer(prediction)
    refs = [normalize_vqa_answer(r) for r in references]
    n = len(refs)
    return sum(min(1.0, sum(1 for j in range(n) if j != i and refs[j] == pred) / 3.0) for i in range(n)) / n


def gqa_accuracy(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_vqa_answer(prediction)
    return 1.0 if pred and any(pred == normalize_vqa_answer(r) for r in references) else 0.0


def _first_integer(text: str) -> Optional[str]:
    for tok in normalize_vqa_answer(text).split():
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            return str(int(tok))
    return None


def count_accuracy(prediction: str, references: Sequence[str]) -> float:
    """TallyQA exact-count accuracy."""
    pred = _first_integer(prediction)
    return 1.0 if pred is not None and any(_first_integer(r) == pred for r in references) else 0.0


_YES = {"yes", "true", "y", "t", "1", "correct"}
_NO = {"no", "false", "n", "f", "0", "incorrect"}


def _yes_no(text: str) -> Optional[str]:
    norm = _anls_normalize(text)
    if not norm:
        return None
    head = norm.split(" ", 1)[0].rstrip(".,!?")
    return "yes" if head in _YES else "no" if head in _NO else None


def pope_accuracy(prediction: str, references: Sequence[str]) -> float:
    pred = _yes_no(prediction)
    return 1.0 if pred is not None and any(_yes_no(r) == pred for r in references) else 0.0


# --------------------------------------------------------------------------- #
# RefCOCO+ (Acc@IoU >= 0.5)
# --------------------------------------------------------------------------- #
Box = Tuple[float, float, float, float]
_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]")

#: Prompt for referring-expression grounding; boxes are normalised XYXY in [0, 1].
REC_PROMPT = (
    "Bounding box coordinates are specified in the format (top-left x, top-left y, bottom-right x, "
    "bottom-right y). All values are floating point numbers bounded between 0 and 1. Please provide "
    "the bounding box coordinate of the region this sentence describes: {expression}"
)


def parse_box(text: str) -> Optional[Box]:
    """First ``[x1, y1, x2, y2]`` found in the text."""
    m = _BOX_RE.search(text or "")
    return None if m is None else tuple(float(m.group(i)) for i in range(1, 5))  # type: ignore[return-value]


def format_box(box: Sequence[float]) -> str:
    """Target string for a normalised box, e.g. ``[0.125, 0.250, 0.500, 0.750]``."""
    return "[" + ", ".join(f"{float(v):.3f}" for v in box[:4]) + "]"


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def grounding_accuracy(prediction: str, references: Sequence[str], threshold: float = 0.5) -> float:
    """1 if IoU(predicted box, reference box) >= threshold."""
    pred = parse_box(prediction)
    if pred is None:
        return 0.0
    best = 0.0
    for r in references:
        gt = parse_box(r)
        if gt is not None:
            best = max(best, iou(pred, gt))
    return 1.0 if best >= threshold else 0.0


# --------------------------------------------------------------------------- #
METRICS: Dict[str, Callable[[str, Sequence[str]], float]] = {
    "anls": anls,
    "relaxed_accuracy": relaxed_accuracy,
    "vqa_accuracy": vqa_accuracy,
    "gqa_accuracy": gqa_accuracy,
    "count_accuracy": count_accuracy,
    "pope_accuracy": pope_accuracy,
    "acc_iou50": grounding_accuracy,
}


def score(metric: str, prediction: str, references: Sequence[str]) -> float:
    """Per-example score in [0, 1] for one of :data:`METRICS`."""
    return METRICS[metric](prediction, list(references))


def corpus_score(metric: str, predictions: List[str], references: List[Sequence[str]]) -> float:
    """Mean score on a 0-100 scale."""
    if not predictions:
        return 0.0
    return 100.0 * sum(score(metric, p, r) for p, r in zip(predictions, references)) / len(predictions)


__all__ = ["METRICS", "score", "corpus_score", "anls", "relaxed_accuracy", "vqa_accuracy", "gqa_accuracy",
           "count_accuracy", "pope_accuracy", "grounding_accuracy", "normalize_vqa_answer", "parse_box",
           "format_box", "iou", "REC_PROMPT"]
