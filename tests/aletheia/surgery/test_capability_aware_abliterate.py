"""CapabilityAwareAbliterator tests."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

torch = pytest.importorskip("torch")

from kairos.aletheia.surgery.capability_aware_abliterate import (
    CapabilityAwareAbliterator,
)
from kairos.aletheia.surgery.refusal_direction import (
    compute_direction_from_activations,
    compute_capability_subspace,
)


def _build_inputs(d=16):
    """Generate harmful, harmless, axis-i and neutral activations such
    that the refusal direction lies partially along axis 0 (math) and
    partially along an orthogonal direction."""
    torch.manual_seed(0)
    neutral = torch.zeros(20, d)
    harmless = torch.zeros(20, d)
    # Harmful prompts produce activations with components on axis 0
    # (capability) AND axis 5 (orthogonal-to-capability):
    harmful = torch.zeros(20, d)
    harmful[:, 0] = 0.6
    harmful[:, 5] = 0.8
    axis_math = torch.zeros(20, d)
    axis_math[:, 0] = 1.0
    return dict(
        neutral=neutral,
        harmless=harmless,
        harmful=harmful,
        axes={"math": axis_math},
    )


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


def test_prepare_orthogonalises_refusal():
    inputs = _build_inputs()
    r = compute_direction_from_activations(inputs["harmful"], inputs["harmless"])
    cap = compute_capability_subspace(inputs["axes"], inputs["neutral"])
    abl = CapabilityAwareAbliterator(r, cap)
    rep = abl.prepare()
    # After orthogonalisation, r_pure should have NO component on axis 0
    assert abs(float(abl.r_pure[0])) < 1e-5
    # And should be unit norm
    assert abs(float(abl.r_pure.norm()) - 1.0) < 1e-5
    # Per-axis overlap should be non-zero on "math"
    assert "math" in rep.axis_overlaps
    assert abs(rep.axis_overlaps["math"]) > 0.1


def test_prepare_full_overlap_when_r_in_capability():
    """If r lies fully in the capability subspace, r_pure should have
    near-zero norm."""
    d = 8
    h = torch.zeros(10, d)
    h[:, 0] = 1.0
    b = torch.zeros(10, d)
    axis = torch.zeros(10, d)
    axis[:, 0] = 1.0
    neut = torch.zeros(10, d)
    r = compute_direction_from_activations(h, b)
    cap = compute_capability_subspace({"x": axis}, neut)
    abl = CapabilityAwareAbliterator(r, cap)
    rep = abl.prepare()
    assert rep.refusal_norm_after_orthogonalise < 1e-5


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


def test_apply_projects_targeted_layers_only():
    """Only weights matching target_suffixes should be touched."""
    inputs = _build_inputs()
    r = compute_direction_from_activations(inputs["harmful"], inputs["harmless"])
    cap = compute_capability_subspace(inputs["axes"], inputs["neutral"])
    abl = CapabilityAwareAbliterator(r, cap)

    d = r.direction.shape[0]
    state = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(d, 12),
        "model.layers.0.mlp.down_proj.weight":   torch.randn(d, 12),
        "model.layers.0.mlp.gate_proj.weight":   torch.randn(d, 12),  # not targeted
        "model.layers.0.input_layernorm.weight": torch.ones(d),       # 1D, skipped
    }
    out = abl.apply(state)
    # o_proj and down_proj should differ
    for k in ["model.layers.0.self_attn.o_proj.weight",
              "model.layers.0.mlp.down_proj.weight"]:
        diff = (out[k].float() - state[k].float()).norm().item()
        assert diff > 1e-4, f"{k} was not touched"
    # gate_proj and layernorm should be unchanged
    for k in ["model.layers.0.mlp.gate_proj.weight",
              "model.layers.0.input_layernorm.weight"]:
        assert torch.allclose(out[k].float(), state[k].float())
    assert abl.report.n_touched == 2


def test_apply_skips_vision_audio_paths():
    """Skip-substring filter prevents touching vision / audio towers."""
    inputs = _build_inputs()
    r = compute_direction_from_activations(inputs["harmful"], inputs["harmless"])
    cap = compute_capability_subspace(inputs["axes"], inputs["neutral"])
    abl = CapabilityAwareAbliterator(r, cap)

    d = r.direction.shape[0]
    state = {
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(d, 8),
        "model.vision_tower.layers.0.o_proj.weight":             torch.randn(d, 8),
    }
    abl.apply(state)
    assert abl.report.n_touched == 1
    assert "model.language_model.layers.0.self_attn.o_proj.weight" in abl.report.touched_layers
    assert abl.report.n_skipped == 1


def test_apply_projection_removes_refusal_direction_from_output():
    """For a touched layer, the column-space of W' should be orthogonal
    to r_pure: r_pureᵀ · W' ≈ 0."""
    inputs = _build_inputs()
    r = compute_direction_from_activations(inputs["harmful"], inputs["harmless"])
    cap = compute_capability_subspace(inputs["axes"], inputs["neutral"])
    abl = CapabilityAwareAbliterator(r, cap)
    abl.prepare()

    d = abl.r_pure.shape[0]
    w = torch.randn(d, 8)
    state = {"o_proj.weight": w}
    out = abl.apply(state)
    projected_output = abl.r_pure @ out["o_proj.weight"].float()
    assert float(projected_output.norm().item()) < 1e-4


def test_export_codebook_contains_expected_fields():
    inputs = _build_inputs()
    r = compute_direction_from_activations(inputs["harmful"], inputs["harmless"])
    cap = compute_capability_subspace(inputs["axes"], inputs["neutral"])
    abl = CapabilityAwareAbliterator(r, cap)
    abl.prepare()
    book = abl.export_codebook()
    for key in ("r_pure", "capability_basis", "capability_axes",
                  "axis_overlaps", "touched_layers"):
        assert key in book


def test_in_place_mutation():
    inputs = _build_inputs()
    r = compute_direction_from_activations(inputs["harmful"], inputs["harmless"])
    cap = compute_capability_subspace(inputs["axes"], inputs["neutral"])
    abl = CapabilityAwareAbliterator(r, cap)
    d = r.direction.shape[0]
    state = {"o_proj.weight": torch.randn(d, 4)}
    original = state["o_proj.weight"].clone()
    out = abl.apply(state, in_place=True)
    assert out is state
    assert not torch.allclose(state["o_proj.weight"].float(), original)
