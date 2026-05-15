"""Tests for v0.3.0 additions:

- KairosAntiResonantInit (qGPT-Infinity port)
- recommended_bundle() one-line factory
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
nn = torch.nn

from kairos import (
    AntiResonantReport,
    CallbackBundle,
    KairosAntiResonantInit,
    KairosCheckpoint,
    KairosEarlyStop,
    KairosPendulumLR,
    KairosParetoGuard,
    recommended_bundle,
)


# ---------------------------------------------------------------------------
# KairosAntiResonantInit
# ---------------------------------------------------------------------------


def _make_model(in_dim: int = 32, hidden: int = 64, vocab: int = 100):
    return nn.Sequential(
        nn.Embedding(vocab, in_dim),
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, vocab),
    )


def test_antiresonant_constructs():
    init = KairosAntiResonantInit()
    assert init.suppress_top_k == 8
    assert init.scale_factor == 0.02


def test_antiresonant_invalid_suppress_raises():
    with pytest.raises(ValueError):
        KairosAntiResonantInit(suppress_top_k=-1)


def test_antiresonant_invalid_scale_raises():
    with pytest.raises(ValueError):
        KairosAntiResonantInit(scale_factor=0.0)


def test_antiresonant_dry_run_returns_report():
    m = _make_model()
    init = KairosAntiResonantInit()
    rep = init.apply(m, dry_run=True)
    assert isinstance(rep, AntiResonantReport)
    assert rep.n_linear == 3
    assert rep.n_embedding == 1
    # LayerNorm should NOT be counted in linear/embedding (and not skipped
    # silently — it's a separate skip bucket for "module-with-params").
    assert rep.n_skipped >= 1


def test_antiresonant_caps_spectral_norm_without_teacher():
    m = _make_model()
    init = KairosAntiResonantInit(scale_factor=0.05, suppress_top_k=0,
                                    seed=0)
    init.apply(m)
    # All linear weight spectral norms should be ~ 0.05
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            S = torch.linalg.svdvals(mod.weight)
            assert float(S.max().item()) < 0.05 + 1e-3, (
                f"layer {name}: top sv = {float(S.max())}, expected ~0.05"
            )


def test_antiresonant_embedding_phase_staggered():
    m = nn.Embedding(20, 16)
    init = KairosAntiResonantInit(embedding_scale=0.1,
                                    phase_staggered_embeddings=True,
                                    seed=0)
    init.apply(nn.Sequential(m))
    # Row max should equal embedding_scale within tolerance
    row_max = m.weight.detach().abs().amax(dim=1)
    # Last row's cosine phase can have a zero crossing — but row max
    # of cos over a full row of length 16 ≥ 0.95 amplitude before
    # rescaling, which is then rescaled to embedding_scale=0.1.
    assert float(row_max.max().item()) <= 0.1 + 1e-5
    assert float(row_max.min().item()) >= 0.1 - 1e-5


def test_antiresonant_suppresses_teacher_top_k():
    """When teacher is supplied, student's top-K should NOT align
    with teacher's top-K (subspace projection)."""
    torch.manual_seed(0)
    teacher = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 32))
    # Initialize teacher with structured weights (top-K dominant)
    with torch.no_grad():
        U, _ = torch.linalg.qr(torch.randn(64, 64))
        V, _ = torch.linalg.qr(torch.randn(32, 32))
        sv = torch.tensor([10.0, 8.0, 6.0, 4.0] + [0.1] * 28)
        teacher[0].weight.copy_(U[:, :32] @ torch.diag(sv) @ V.T)

    student = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 32))
    init = KairosAntiResonantInit(suppress_top_k=4, scale_factor=0.05,
                                    seed=42)
    rep = init.apply(student, teacher=teacher)

    assert rep.suppressed_directions >= 4
    # Check overlap: student's top singular vector should have small
    # alignment with teacher's top-4 subspace.
    U_t, _, _ = torch.linalg.svd(teacher[0].weight.float(),
                                   full_matrices=False)
    top_subspace = U_t[:, :4]  # (64, 4)
    U_s, _, _ = torch.linalg.svd(student[0].weight.float(),
                                   full_matrices=False)
    # Project student's top-1 sv onto teacher's top-4 subspace
    proj = top_subspace.T @ U_s[:, :1]
    overlap = float(proj.norm().item())
    # Random alignment for an isotropic vector in 64-dim with 4-dim
    # subspace is sqrt(4/64) = 0.25. Suppression should give us <<0.25.
    assert overlap < 0.15, (
        f"student top-1 should be anti-aligned with teacher top-4, "
        f"got overlap={overlap:.3f}"
    )


# ---------------------------------------------------------------------------
# recommended_bundle
# ---------------------------------------------------------------------------


class _FakeOptimizer:
    def __init__(self, lr: float = 1e-3) -> None:
        self.param_groups = [{"lr": float(lr), "weight_decay": 0.0}]


def test_recommended_bundle_grokking():
    opt = _FakeOptimizer()
    bundle = recommended_bundle("grokking", optimizer=opt,
                                 max_steps=10000)
    assert isinstance(bundle, CallbackBundle)
    types = {type(cb).__name__ for cb in bundle.callbacks}
    assert "KairosPendulumLR" in types
    assert "KairosEarlyStop" in types


def test_recommended_bundle_grokking_pendulum_uses_test_loss():
    opt = _FakeOptimizer()
    bundle = recommended_bundle("grokking", optimizer=opt,
                                 max_steps=1000)
    pend = next(cb for cb in bundle.callbacks
                  if type(cb).__name__ == "KairosPendulumLR")
    assert pend.metric == "test_loss"


def test_recommended_bundle_distillation():
    opt = _FakeOptimizer()
    bundle = recommended_bundle("distillation", optimizer=opt)
    pend = next(cb for cb in bundle.callbacks
                  if type(cb).__name__ == "KairosPendulumLR")
    assert pend.metric == "train_loss"


def test_recommended_bundle_pareto_requires_anchor():
    with pytest.raises(ValueError):
        recommended_bundle("pareto_post_training")


def test_recommended_bundle_pareto_with_anchor():
    bundle = recommended_bundle(
        "pareto_post_training",
        anchor={"acc": 0.5, "cal": 0.7},
    )
    types = {type(cb).__name__ for cb in bundle.callbacks}
    assert "KairosParetoGuard" in types


def test_recommended_bundle_growth_search():
    opt = _FakeOptimizer()
    bundle = recommended_bundle("growth_search", optimizer=opt)
    types = {type(cb).__name__ for cb in bundle.callbacks}
    assert "KairosGrowthController" in types


def test_recommended_bundle_pretraining():
    opt = _FakeOptimizer()
    bundle = recommended_bundle("pretraining", optimizer=opt)
    types = {type(cb).__name__ for cb in bundle.callbacks}
    assert "KairosPendulumLR" in types
    pend = next(cb for cb in bundle.callbacks
                  if type(cb).__name__ == "KairosPendulumLR")
    # Pretraining uses train_loss (monotone descent, like distillation)
    assert pend.metric == "train_loss"


def test_recommended_bundle_with_checkpoint_dir(tmp_path):
    opt = _FakeOptimizer()
    bundle = recommended_bundle("grokking", optimizer=opt,
                                 max_steps=1000,
                                 checkpoint_dir=str(tmp_path))
    types = {type(cb).__name__ for cb in bundle.callbacks}
    assert "KairosCheckpoint" in types


def test_recommended_bundle_unknown_profile_raises():
    with pytest.raises(ValueError):
        recommended_bundle("bogus_profile")


def test_recommended_bundle_observe_smoke():
    """End-to-end: bundle must accept a metric stream without crashing."""
    opt = _FakeOptimizer()
    bundle = recommended_bundle("grokking", optimizer=opt, max_steps=200)
    for step in range(50):
        bundle.observe(step, train_loss=0.5, train_acc=0.5,
                        test_loss=0.5, test_acc=0.3)
