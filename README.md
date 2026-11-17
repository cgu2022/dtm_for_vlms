# DTM for Vision-Language Models: Supplementary Code

This folder contains code for the method, baselines, training and evaluation described in the paper:
- **DTM framework.** A learned delete gate, soft-to-hard deletion and a PID-controlled deletion objective.
- **Two instances.**
  - **V-MrT5Gemma2**: DTM in the encoder of T5Gemma 2.
  - **V-MrGemma3**: DTM during prompt prefill in Gemma 3.
- **Baselines.** FastV, φ-R, Random and Base (no deletion).
- **Evaluation.** Task metrics, analytical FLOPs accounting, and the spatial evidence-retention analysis.

## Folder map

```
dtm/                       model-agnostic DTM components
  config.py                DeletionConfig: method, deletion layer, delta, c, tau, aggregation
  gate.py                  delete gate (Eq. 1) and initial deletion-rate calibration (App. A.2)
  attention.py             softmax_1 soft deletion (Eq. 2) and answer-query biases (Sec. 4.3)
  deletion.py              hard deletion: natural threshold, exact budget, compaction
  objective.py             deletion regularizer L_G and observed rate r_t
  controller.py            PID controller for alpha_t (Eq. 3)
  baselines.py             FastV / phi-R / Random selection scores and query aggregation
  models/t5gemma2.py       V-MrT5Gemma2 (+ baselines) on T5Gemma 2
  models/gemma3.py         V-MrGemma3 (+ baselines) on Gemma 3
training/
  train.py                 fine-tuning entry point (LoRA + gate), paper hyperparameters as defaults
  trainer.py               loss L = L_CE + alpha_t L_G, calibration, PID updates, optimizer groups
  data.py                  prompts, targets, image preprocessing, collator
evaluation/
  evaluate.py              matched-rate and fixed-checkpoint evaluation with exact budgets
  metrics.py               ANLS, relaxed accuracy, VQA accuracy, GQA/TallyQA/POPE accuracy, Acc@IoU>=0.5
  tasks.py                 the ten benchmarks: sources, splits, example counts, metrics
  flops.py                 analytical FLOPs and savings (App. G)
analysis/
  evidence_retention.py    spatial evidence retention (Sec. 6.3)
tests/                     CPU unit tests (tiny random models; no downloads)
```

## Paper-to-code map

| Paper | Code |
|---|---|
| Eq. 1: delete gate `g = c·σ(−(RMSNorm(H)w + b))`, `c = −22` | `dtm/gate.py::DeleteGate` |
| Sec. 4.1: vision-only deletion (text and boundary tokens `g = 0`) | `DeleteGate.forward`; `visual_mask` in `dtm/models/*` |
| App. A.2: initial deletion-rate calibration `b0 = −logit(τ/c) − q0` | `DeleteGate.calibrate_`; called on the first training batch in `training/trainer.py` |
| Eq. 2: soft deletion with `softmax_1` | `dtm/attention.py::softmax_1`, `eager_attention_forward_softmax_1`, `key_bias` |
| Sec. 4.3: answer-query biases `min(g_j − τ, 0)`, `τ = −7` | `dtm/attention.py::answer_query_bias` |
| Sec. 3.2: hard deletion at inference (`g_j ≤ τ`) | `dtm/deletion.py::natural_delete_mask`, `compact` |
| Sec. 5 / App. C: exact budget, delete `⌊δ·n_v⌋` visual tokens | `dtm/deletion.py::exact_budget_delete_mask` |
| Sec. 3.2 / Sec. 4.1: `L_G` and observed rate `r_t` over visual tokens | `dtm/objective.py` |
| Eq. 3: PID controller (`k_p=1, k_i=1e−5, k_d=0.5, ω=10, ρ=0.9, α_0=0.1`) | `dtm/controller.py::PIDController` |
| Sec. 4.1, Fig. 1a: V-MrT5Gemma2 | `dtm/models/t5gemma2.py` |
| Sec. 4.2, Fig. 1b: V-MrGemma3 (prefill deletion, KV-cache compaction) | `dtm/models/gemma3.py` |
| Sec. 5, App. D.6: FastV, φ-R, Random; `all_mean` / `text_mean` / `text_last` | `dtm/baselines.py`; `dtm/models/*` |
| Sec. 5: matched-rate and fixed-checkpoint settings | `evaluation/evaluate.py` |
| App. C: tasks, splits and metrics | `evaluation/tasks.py`, `evaluation/metrics.py` |
| App. G: analytical FLOPs and savings | `evaluation/flops.py` |
| Sec. 6.3: spatial evidence retention | `analysis/evidence_retention.py` |

## Layer convention

As in the paper:
- Layer 0 denotes the input states, and transformer layers are indexed from 1.
- Global-attention layers are 6, 12, 18, ….
- **L5** (the setting of all main experiments) means that the gate, or the attention-based selector, reads the output of layer 5, and layers 6, 7, … process the shortened sequence (`k = 6`).
- In zero-based Python indexing over the layer list, `layers[0:5]` see the full sequence and `layers[5:]` see the shortened one (`DeletionConfig.delete_after_layer = 5`).

## Default hyperparameters (App. C)

| Setting | Value | Where |
|---|---|---|
| Adapter / gate learning rate | 1e-4 / 1e-4 (separate parameter group) | `training/train.py` |
| Schedule | cosine, no warmup | `training/train.py` |
| Weight decay | 1e-4 | `training/train.py` |
| Batch size | 256 (Gemma 3 4B: 128) | `training/train.py` |
| Gradient clipping | max norm 1 | `training/train.py` |
| LoRA | r = 32, α = 64, dropout 0.05; attention Q/K/V/O and MLP gate/up/down | `training/train.py` |
| Vision encoder | frozen (no LoRA) | `training/train.py` |
| Precision | BF16 backbone | `training/train.py` |
| Gate scale / threshold | c = −22 / τ = −7 | `dtm/config.py` |
| PID gains / lookback | k_p = 1, k_i = 1e−5, k_d = 0.5 / ω = 10 | `dtm/controller.py` |
| Deletion layer | L5 (`k = 6`) | `dtm/config.py` |
| Inference budget | exactly `⌊δ·n_v⌋` visual tokens per example | `dtm/deletion.py` |

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q          # CPU only; tiny random models, no downloads
```

## Training and evaluation

```bash
# DTM on T5Gemma 2 1B-1B, DocVQA, target deletion rate 0.6
python -m training.train --backbone t5gemma2-1b-1b --method dtm --task docvqa --delta 0.6 \
    --max_steps 300 --output_dir runs/docvqa_dtm_0.6

# A baseline: FastV with all-query aggregation (text_mean / text_last via --aggregation)
python -m training.train --backbone t5gemma2-1b-1b --method fastv --aggregation all_mean \
    --task docvqa --delta 0.6 --max_steps 300 --output_dir runs/docvqa_fastv_0.6

# Matched-rate evaluation (at the trained rate) and fixed-checkpoint evaluation (10%..90%)
python -m evaluation.evaluate --checkpoint runs/docvqa_dtm_0.6/best --setting matched
python -m evaluation.evaluate --checkpoint runs/docvqa_dtm_0.6/best --setting fixed

# POPE is evaluated with checkpoints fine-tuned on VQAv2
python -m evaluation.evaluate --checkpoint runs/vqav2_dtm_0.6/best --setting matched --task pope
```

Training saves the LoRA adapter, the gate (DTM only) and the deletion settings; `best/` holds the checkpoint with the best validation score at the trained rate. Evaluation writes per-example predictions, scores and FLOPs savings to a JSON file next to the checkpoint.

Datasets are loaded from the Hugging Face Hub with the identifiers in `evaluation/tasks.py`. The RefCOCO+ training export references COCO images by id rather than embedding them; point `COCO_IMAGE_DIR` to a folder of COCO images (`<12-digit id>.jpg`) before training on RefCOCO+.

Backbones: `t5gemma2-1b-1b` (`google/t5gemma-2-1b-1b`), `t5gemma2-270m-270m` (`google/t5gemma-2-270m-270m`), and `gemma3-4b` (`google/gemma-3-4b-pt`). Methods: `dtm`, `fastv`, `phi_r`, `random` and `base`.
