"""The two DTM instances of the paper, each also hosting the baselines (Sec. 5).

* :class:`VMrT5Gemma2ForConditionalGeneration`: V-MrT5Gemma2, deletion inside the
  T5Gemma 2 encoder (Sec. 4.1, Fig. 1a).
* :class:`VMrGemma3ForConditionalGeneration`: V-MrGemma3, deletion during prompt
  prefill of the decoder-only Gemma 3 (Sec. 4.2, Fig. 1b).
"""

from .gemma3 import VMrGemma3ForConditionalGeneration
from .t5gemma2 import VMrT5Gemma2ForConditionalGeneration

__all__ = ["VMrT5Gemma2ForConditionalGeneration", "VMrGemma3ForConditionalGeneration"]
