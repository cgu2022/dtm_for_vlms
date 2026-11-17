"""Unit tests tying the core DTM components to the paper's equations."""

import math

import pytest
import torch

from dtm.attention import answer_query_bias, softmax_1
from dtm.baselines import aggregate_received_attention, query_rows, random_scores
from dtm.config import DeletionConfig
from dtm.controller import PIDConfig, PIDController
from dtm.deletion import compact, exact_budget_delete_mask, natural_delete_mask
from dtm.gate import DeleteGate
from dtm.objective import gate_loss, observed_rate


def _batch(b=3, n=40, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(b, n, d, generator=g)
    vis = torch.zeros(b, n, dtype=torch.bool)
    vis[:, 2:34] = True  # 32 visual tokens after a boundary marker
    return h, vis


def test_gate_parameters_and_range():
    """Eq. 1: 2d+1 parameters, g in (c, 0) on visual tokens, g = 0 elsewhere."""
    h, vis = _batch()
    gate = DeleteGate(32, c=-22.0, tau=-7.0)
    assert sum(p.numel() for p in gate.parameters()) == 2 * 32 + 1
    g = gate(h, vis)
    assert torch.all(g[vis] < 0) and torch.all(g[vis] > -22.0)
    assert torch.all(g[~vis] == 0)


def test_calibration_hits_initial_rate():
    """App. A.2: after calibration, g_j <= tau exactly when z_j <= q0."""
    h, vis = _batch()
    gate = DeleteGate(32)
    for delta_init in (0.2, 0.4, 0.6, 0.8):
        gate.calibrate_(h, vis, delta_init)
        g = gate(h, vis)
        z = gate.projection(h[vis])
        q0 = torch.quantile(z, delta_init)
        assert torch.equal(g[vis] <= gate.tau + 1e-4, z <= q0 + 1e-6)
        rate = observed_rate(g, vis, gate.tau)
        assert abs(rate - delta_init) <= 1.0 / vis.sum().item() + 1e-6
    assert bool(gate.calibrated)


def test_exact_budget_counts_with_ties():
    """Every method deletes exactly floor(delta * n_v) visual tokens, even with tied scores."""
    _, vis = _batch()
    vis[1, 30:34] = False  # rows with different n_v
    for delta in (0.1, 0.2, 0.5, 0.6, 0.8, 0.9):
        for scores in (torch.randn(vis.shape), torch.zeros(vis.shape)):
            m = exact_budget_delete_mask(scores, vis, delta)
            n_v = vis.sum(1)
            assert torch.equal(m.sum(1), torch.floor(n_v.float() * delta).long())
            assert not (m & ~vis).any()  # text is never deleted


def test_natural_deletion_and_compaction_keep_positions():
    h, vis = _batch()
    g = torch.full(vis.shape, -1.0)
    g[:, 5:10] = -10.0
    delete = natural_delete_mask(g, vis, -7.0)
    assert delete.sum().item() == 3 * 5
    packed, index, valid = compact(h, ~delete)
    assert packed.shape[1] == 35 and valid.all()
    assert torch.equal(index[0, :5], torch.arange(5))  # order and original positions kept
    assert torch.equal(index[0, 5], torch.tensor(10))
    assert torch.equal(packed, torch.gather(h, 1, index[..., None].expand(-1, -1, 32)))


def test_softmax_1():
    """Eq. 2: A_ij = exp(s_ij) / (1 + sum_u exp(s_iu))."""
    s = torch.randn(2, 3, 4, 5)
    ref = s.exp() / (1 + s.exp().sum(-1, keepdim=True))
    assert torch.allclose(softmax_1(s), ref, atol=1e-6)
    # a uniformly suppressed row dissipates its mass
    assert softmax_1(torch.full((1, 4), -60.0)).sum() < 1e-20


def test_answer_query_bias():
    """Sec. 4.3: min(g - tau, 0) on visual keys, 0 on text keys."""
    g = torch.tensor([[-3.0, -7.0, -10.0, 0.0]])
    vis = torch.tensor([[True, True, True, False]])
    assert torch.equal(answer_query_bias(g, vis, -7.0), torch.tensor([[0.0, 0.0, -3.0, 0.0]]))


def test_gate_loss_over_visual_tokens():
    g = torch.tensor([[-1.0, -3.0, 0.0], [-5.0, 0.0, 0.0]])
    vis = torch.tensor([[True, True, False], [True, False, False]])
    assert gate_loss(g, vis).item() == pytest.approx(-3.0)
    assert observed_rate(g, vis, -2.0) == pytest.approx(2 / 3)


def test_pid_matches_eq3():
    """Eq. 3 by hand, including the EMA P term, the D lookback and the clamp."""
    cfg = PIDConfig()
    ctl = PIDController(0.6, cfg)
    assert ctl.alpha == cfg.alpha_0
    rates = [0.2, 0.3, 0.4, 0.5, 0.55, 0.58, 0.6, 0.62, 0.61, 0.6, 0.6, 0.59, 0.9]
    ebar, integ, errs = 0.0, 0.0, []
    for r in rates:
        e = 0.6 - r
        ebar = cfg.rho * ebar + (1 - cfg.rho) * e
        integ += e
        errs.append(e)
        d = cfg.k_d * (errs[-1] - errs[-1 - cfg.omega]) if len(errs) > cfg.omega else 0.0
        expected = max(0.0, cfg.k_p * ebar + cfg.k_i * integ + d)
        assert ctl.step(r) == pytest.approx(expected, abs=1e-12)
    assert ctl.alpha >= 0.0


def test_query_aggregation():
    attn = torch.softmax(torch.randn(2, 4, 6, 6), -1)
    valid = torch.ones(2, 6, dtype=torch.bool)
    text = torch.tensor([[False, False, False, True, True, False]] * 2)
    assert torch.allclose(aggregate_received_attention(attn, valid, text, "all_mean"), attn.mean(dim=(1, 2)))
    assert torch.allclose(aggregate_received_attention(attn, valid, text, "text_mean"), attn[:, :, 3:5].mean(dim=(1, 2)))
    assert torch.allclose(aggregate_received_attention(attn, valid, text, "text_last"), attn[:, :, 4].mean(dim=1))
    assert query_rows(valid, text, "text_last").sum().item() == 2


def test_random_scores_are_seeded():
    _, vis = _batch()
    a = random_scores(vis, torch.Generator().manual_seed(1))
    b = random_scores(vis, torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_config_defaults():
    cfg = DeletionConfig()
    assert (cfg.c, cfg.tau, cfg.delete_after_layer) == (-22.0, -7.0, 5)
    assert cfg.use_softmax_1 and not DeletionConfig(method="fastv").use_softmax_1
    assert cfg.calibration_rate == cfg.delta
    with pytest.raises(ValueError):
        DeletionConfig(c=-5.0, tau=-7.0)
