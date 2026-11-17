"""Deletion settings shared by every model in this package.

Paper references:
  * Sec. 3.2 / Eq. 1: gate scale ``c`` and threshold ``tau`` (``c < tau < 0``).
  * Sec. 4.3 "Other settings": ``c = -22`` and ``tau = -7`` throughout.
  * Sec. 5 "Baselines": every method deletes visual tokens after five layers
    (L5, i.e. ``k = 6``).

Layer convention (Sec. 3.1, App. B.3): layer 0 denotes the input states and
transformer layers are indexed from 1. ``delete_after_layer = 5`` ("L5")
means the gate reads the output of layer 5 and layers 6, 7, ... process the
shortened sequence (``k = 6``). In zero-based Python indexing over
``model.layers`` this is: ``layers[0:5]`` see the full sequence and
``layers[5:]`` see the shortened one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Method = Literal["dtm", "fastv", "phi_r", "random", "base"]
Aggregation = Literal["all_mean", "text_mean", "text_last"]

METHODS: tuple[str, ...] = ("dtm", "fastv", "phi_r", "random", "base")
AGGREGATIONS: tuple[str, ...] = ("all_mean", "text_mean", "text_last")


@dataclass
class DeletionConfig:
    """Which token-deletion method a model runs, and where.

    Attributes:
        method: ``"dtm"`` (learned delete gate), ``"fastv"``, ``"phi_r"``,
            ``"random"``, or ``"base"`` (no deletion).
        delete_after_layer: number of full-sequence layers before deletion
            (the paper's "L5" is 5, i.e. ``k = 6``).
        delta: target deletion rate used during training (fraction of visual
            tokens removed).
        delta_init: initial deletion rate for the one-shot gate-bias
            calibration (App. A.2); ``None`` uses ``delta``.
        c: gate scale (Eq. 1), ``c < 0``.
        tau: deletion threshold, ``c < tau < 0``.
        aggregation: query aggregation for FastV / phi-R (Sec. 5, App. D.6).
        softmax_1: attention normaliser; ``None`` means softmax_1 for DTM
            (Eq. 2) and the backbone's standard softmax for every other method.
        seed: seed for Random deletion.
    """

    method: Method = "dtm"
    delete_after_layer: int = 5
    delta: float = 0.6
    delta_init: Optional[float] = None
    c: float = -22.0
    tau: float = -7.0
    aggregation: Aggregation = "all_mean"
    softmax_1: Optional[bool] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}")
        if not (self.c < self.tau < 0.0):
            raise ValueError(f"need c < tau < 0, got c={self.c}, tau={self.tau}")
        if not (0.0 <= self.delta <= 1.0):
            raise ValueError(f"delta must be in [0, 1], got {self.delta}")
        if self.delete_after_layer < 1:
            raise ValueError("delete_after_layer must be >= 1")

    @property
    def use_softmax_1(self) -> bool:
        """softmax_1 for DTM (Eq. 2); standard softmax otherwise."""
        if self.softmax_1 is not None:
            return bool(self.softmax_1)
        return self.method == "dtm"

    @property
    def calibration_rate(self) -> float:
        """delta_init for the gate-bias calibration (App. A.2)."""
        return float(self.delta if self.delta_init is None else self.delta_init)

    @property
    def deletes(self) -> bool:
        return self.method != "base"
