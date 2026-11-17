"""DTM: Dynamic Token Merging for vision-language models.

Model-agnostic components of the DTM framework (Secs. 3.2 and 4 of the paper):

* :mod:`dtm.gate`       learned delete gate (Eq. 1) and bias calibration (App. A.2)
* :mod:`dtm.attention`  soft deletion with softmax_1 (Eq. 2), answer-query biases (Sec. 4.3)
* :mod:`dtm.deletion`   hard deletion: natural threshold, exact budget, compaction
* :mod:`dtm.objective`  deletion regularizer L_G and observed rate r_t
* :mod:`dtm.controller` PID control of alpha_t (Eq. 3)
* :mod:`dtm.baselines`  FastV, phi-R and Random selection scores (Sec. 5)
* :mod:`dtm.models`     V-MrT5Gemma2 (T5Gemma 2) and V-MrGemma3 (Gemma 3)
"""

from .config import DeletionConfig
from .controller import PIDConfig, PIDController
from .gate import DeleteGate

__all__ = ["DeletionConfig", "DeleteGate", "PIDConfig", "PIDController"]
