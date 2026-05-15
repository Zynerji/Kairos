"""WeightDeltaCodebook tests on synthetic abliteration."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

torch = pytest.importorskip("torch")

from kairos.aletheia.surgery.codebook import WeightDeltaCodebook
from kairos.aletheia.surgery.refusal_direction import (
    compute_capability_subspace,
)


def _make_synthetic_pair(d_out=16, d_in=12, n_layers=3, *, seed=0):
    """Build a synthetic (original, abliterated) state-dict pair where
    abliteration projected out a single rank-1 direction ``r`` from
    every ``o_proj.weight``.

    Returns: (original_sd, abliterated_sd, r_used)
    """
    torch.manual_seed(seed)
    r = torch.randn(d_out)
    r = r / r.norm()
    P = torch.outer(r, r)               # rank-1 projector
    orig: dict = {}
    abl: dict = {}
    for i in range(n_layers):
        w = torch.randn(d_out, d_in)
        orig[f"model.layers.{i}.self_attn.o_proj.weight"] = w
        abl[f"model.layers.{i}.self_attn.o_proj.weight"] = w - P @ w
        # Untouched layer (bias, norm) — should be skipped
        orig[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(d_out)
        abl[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(d_out)
    return orig, abl, r


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def test_codebook_recovers_rank1_direction():
    orig, abl, r = _make_synthetic_pair()
    book = WeightDeltaCodebook()
    rep = book.build(orig, abl)
    assert rep.n_paired == 3            # 3 o_proj layers diff'd
    # Each rank-1 fraction ~ 1.0
    for entry in rep.layer_deltas:
        assert entry.rank1_fraction > 0.999
    # Recovered u should align (up to sign) with the true r
    for entry in rep.layer_deltas:
        alignment = abs(float((entry.u @ r).item()))
        assert alignment > 0.999


def test_codebook_skips_untouched_2d_weights():
    """A 2D weight that abliteration didn't touch should NOT appear in
    the codebook (delta == 0)."""
    orig, abl, _r = _make_synthetic_pair(n_layers=1)
    # Add a 2D weight that's identical in both
    same = torch.randn(8, 8)
    orig["model.layers.0.mlp.up_proj.weight"] = same
    abl["model.layers.0.mlp.up_proj.weight"] = same.clone()
    book = WeightDeltaCodebook()
    book.build(orig, abl)
    assert "model.layers.0.mlp.up_proj.weight" not in book.layers()


def test_codebook_keep_full_delta_flag():
    orig, abl, _r = _make_synthetic_pair(n_layers=1)
    book = WeightDeltaCodebook(keep_full_delta=True)
    book.build(orig, abl)
    for entry in book.report.layer_deltas:
        assert entry.full_delta is not None


# ---------------------------------------------------------------------------
# split_against_capability
# ---------------------------------------------------------------------------


def test_split_marks_pure_refusal_layer():
    """If the refusal direction is orthogonal to capability subspace,
    alpha_cap should be ~0."""
    d = 16
    torch.manual_seed(1)
    # Build C spanning the first 3 dims
    C = torch.eye(d)[:, :3]
    # Build refusal direction orthogonal to C
    r = torch.zeros(d)
    r[5] = 1.0
    P = torch.outer(r, r)
    # Build synthetic pair using this r
    w = torch.randn(d, 8)
    orig = {"layers.0.o_proj.weight": w}
    abl = {"layers.0.o_proj.weight": w - P @ w}
    book = WeightDeltaCodebook()
    book.build(orig, abl)

    class _FakeSubspace:
        basis = C
        axis_names = ["x", "y", "z"]
        axis_directions = {}
    splits = book.split_against_capability(_FakeSubspace())
    assert len(splits) == 1
    parts = list(splits.values())[0]
    assert parts["alpha_cap"] < 1e-5


def test_split_marks_capability_aligned_layer():
    """If refusal direction lies in the capability subspace, alpha_cap
    should be ~1 — abliteration would have damaged capability heavily."""
    d = 8
    C = torch.eye(d)[:, :3]
    r = torch.zeros(d)
    r[1] = 1.0                  # axis 1 is in C
    P = torch.outer(r, r)
    w = torch.randn(d, 4)
    orig = {"o_proj.weight": w}
    abl = {"o_proj.weight": w - P @ w}
    book = WeightDeltaCodebook()
    book.build(orig, abl)

    class _Sub:
        basis = C
    parts = list(book.split_against_capability(_Sub()).values())[0]
    assert parts["alpha_cap"] > 0.99


# ---------------------------------------------------------------------------
# apply_restoration
# ---------------------------------------------------------------------------


def test_alpha_zero_returns_abliterated_unchanged():
    orig, abl, _r = _make_synthetic_pair()
    cap = _identity_capability(d=16, k=3)
    book = WeightDeltaCodebook()
    book.build(orig, abl)
    healed = book.apply_restoration(abl, cap, alpha=0.0)
    for k, v in abl.items():
        assert torch.allclose(healed[k].float(), v.float(), atol=1e-6)


def test_alpha_one_fully_capability_restores_original():
    """When the refusal direction is fully inside the capability
    subspace AND alpha=1, restoration should recover the original
    weights (capability fraction == full delta)."""
    d = 12
    C = torch.eye(d)[:, :4]
    r = torch.zeros(d)
    r[2] = 1.0                  # inside C
    P = torch.outer(r, r)
    w = torch.randn(d, 8)
    orig = {"o_proj.weight": w}
    abl = {"o_proj.weight": w - P @ w}
    book = WeightDeltaCodebook()
    book.build(orig, abl)

    class _Sub:
        basis = C
    healed = book.apply_restoration(abl, _Sub(), alpha=1.0)
    assert torch.allclose(healed["o_proj.weight"].float(), w, atol=1e-5)


def test_alpha_invalid_raises():
    orig, abl, _r = _make_synthetic_pair()
    cap = _identity_capability(d=16, k=3)
    book = WeightDeltaCodebook()
    book.build(orig, abl)
    with pytest.raises(ValueError):
        book.apply_restoration(abl, cap, alpha=1.5)
    with pytest.raises(ValueError):
        book.apply_restoration(abl, cap, alpha=-0.1)


def _identity_capability(d, k):
    class _Sub:
        basis = torch.eye(d)[:, :k]
        axis_names = [f"a{i}" for i in range(k)]
        axis_directions = {}
    return _Sub()


# ---------------------------------------------------------------------------
# inspection helpers
# ---------------------------------------------------------------------------


def test_non_rank1_layers_detection():
    """If the abliteration was rank-2, the rank1_fraction should drop
    below threshold and the layer should be flagged."""
    d = 16
    torch.manual_seed(2)
    # Two orthogonal projectors -> rank-2 delta
    r1 = torch.zeros(d); r1[0] = 1.0
    r2 = torch.zeros(d); r2[1] = 1.0
    P = torch.outer(r1, r1) + torch.outer(r2, r2)
    w = torch.randn(d, 8)
    orig = {"o_proj.weight": w}
    abl = {"o_proj.weight": w - P @ w}
    book = WeightDeltaCodebook(rank1_threshold=0.95)
    book.build(orig, abl)
    flagged = book.non_rank1_layers()
    assert "o_proj.weight" in flagged
