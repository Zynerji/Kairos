"""Math tests for refusal-direction primitives."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

torch = pytest.importorskip("torch")

from kairos.aletheia.surgery.refusal_direction import (
    compute_direction_from_activations,
    compute_capability_subspace,
    project_out_subspace,
)


# ---------------------------------------------------------------------------
# compute_direction_from_activations
# ---------------------------------------------------------------------------


def test_direction_is_diff_of_means_normalised():
    torch.manual_seed(0)
    h = torch.randn(20, 16)
    b = torch.randn(20, 16)
    expected = h.float().mean(0) - b.float().mean(0)
    expected = expected / expected.norm()
    out = compute_direction_from_activations(h, b)
    assert torch.allclose(out.direction, expected, atol=1e-6)
    assert abs(float(out.direction.norm()) - 1.0) < 1e-6


def test_direction_unnormalised():
    h = torch.tensor([[2.0, 0.0]])
    b = torch.tensor([[0.0, 0.0]])
    out = compute_direction_from_activations(h, b, normalise=False)
    assert torch.allclose(out.direction, torch.tensor([2.0, 0.0]))


def test_direction_3d_activations_get_pooled_over_time():
    # (N, T, D) — should be reduced to (N, D) by mean over T
    h = torch.zeros(4, 10, 8)
    h[:, :, 0] = 1.0
    b = torch.zeros(4, 10, 8)
    out = compute_direction_from_activations(h, b)
    # Direction should point along axis 0
    assert abs(float(out.direction[0]) - 1.0) < 1e-6


def test_direction_dim_mismatch_raises():
    h = torch.randn(5, 16)
    b = torch.randn(5, 8)
    with pytest.raises(ValueError):
        compute_direction_from_activations(h, b)


# ---------------------------------------------------------------------------
# compute_capability_subspace
# ---------------------------------------------------------------------------


def test_capability_basis_is_orthonormal():
    d = 32
    torch.manual_seed(0)
    neut = torch.zeros(10, d)
    axes = {
        "x": torch.randn(20, d),
        "y": torch.randn(20, d),
        "z": torch.randn(20, d),
    }
    sub = compute_capability_subspace(axes, neut)
    C = sub.basis
    # Orthonormal: Cᵀ·C = I
    gram = C.t() @ C
    eye = torch.eye(gram.shape[0])
    assert torch.allclose(gram, eye, atol=1e-5)
    assert len(sub.axis_names) == 3


def test_capability_basis_respects_max_rank():
    d = 16
    neut = torch.zeros(5, d)
    axes = {f"axis_{i}": torch.randn(10, d) for i in range(5)}
    sub = compute_capability_subspace(axes, neut, max_rank=2)
    assert sub.basis.shape == (d, 2)
    assert len(sub.axis_names) == 2


def test_capability_degenerate_axis_skipped():
    d = 8
    neut = torch.zeros(5, d)
    axes = {
        "good": torch.randn(10, d) + 5.0,
        "degenerate": torch.zeros(10, d),       # mean == neutral → c == 0
    }
    sub = compute_capability_subspace(axes, neut)
    assert "degenerate" not in sub.axis_names
    assert "good" in sub.axis_names


def test_capability_all_degenerate_raises():
    d = 4
    neut = torch.ones(5, d)
    axes = {"a": torch.ones(10, d)}
    with pytest.raises(ValueError):
        compute_capability_subspace(axes, neut)


# ---------------------------------------------------------------------------
# project_out_subspace
# ---------------------------------------------------------------------------


def test_project_out_orthogonal_to_subspace():
    """After projection, r_pure should be orthogonal to every column of C."""
    d = 16
    torch.manual_seed(0)
    r = torch.randn(d)
    # Build a random 4-dim orthonormal C
    A = torch.randn(d, 4)
    C, _ = torch.linalg.qr(A)
    r_pure = project_out_subspace(r, C)
    # Cᵀ·r_pure should be ~zero
    proj = C.t() @ r_pure
    assert float(proj.norm().item()) < 1e-5


def test_project_out_is_normalised():
    d = 8
    r = torch.randn(d)
    A = torch.randn(d, 2)
    C, _ = torch.linalg.qr(A)
    r_pure = project_out_subspace(r, C)
    assert abs(float(r_pure.norm()) - 1.0) < 1e-5


def test_project_out_returns_zeros_when_r_in_subspace():
    """If r is already inside the capability subspace, r_pure should be
    approximately zero (we don't renormalise the zero vector)."""
    d = 6
    # C spans the first 3 axes
    C = torch.eye(d)[:, :3]
    r = torch.zeros(d)
    r[0] = 1.0                  # fully inside C
    r_pure = project_out_subspace(r, C)
    assert float(r_pure.norm().item()) < 1e-5


def test_project_out_with_namespace_object():
    """Should accept either a raw tensor or a CapabilitySubspace."""
    d = 4
    r = torch.tensor([1.0, 0.0, 0.0, 1.0])
    neut = torch.zeros(5, d)
    axes = {"a": torch.tensor([[0.0, 1.0, 0.0, 0.0]] * 5)}  # diff = [0,1,0,0]
    sub = compute_capability_subspace(axes, neut)
    r_pure = project_out_subspace(r, sub)
    # Subspace is along axis 1; r has no component on axis 1, so r_pure = r/‖r‖
    expected = r / r.norm()
    assert torch.allclose(r_pure, expected, atol=1e-6)
