"""CapabilityAwareAbliterator — Path A: do the abliteration ourselves,
but orthogonalise the refusal direction against a capability subspace
BEFORE projection so capability-correlated content isn't damaged.

Compare against standard (Arditi et al.) abliteration which projects out
the raw refusal direction ``r`` from every weight matrix that writes to
the residual stream. The standard recipe damages any capability that
co-fires with refusal — instruction-following, calibration, sometimes
factuality. Reported costs on huihui-ai / OBLITERATUS abliterations
include 1-5 pp on standard benchmarks plus a 20-30% "soft deflection"
rate where the model nominally complies but produces watered-down
answers.

Capability-aware abliteration:

    r        = compute_refusal_direction(harmful, harmless)
    C        = compute_capability_subspace({math, facts, ...}, neutral)
    r_pure   = r − C·(Cᵀ·r)                # remove capability-aligned component
    r_pure   = r_pure / ‖r_pure‖

    For every W that writes to residual stream:
        W'   = W − r_pure · (r_pureᵀ · W)

The codebook ``(r_pure, C, axis_overlaps={c_i: cᵢᵀ·r})`` is saved with
the result so you have a record of which capability axes the standard
abliteration would have damaged the most.

Targeted layers
===============
By default we touch the same projections the standard recipe does:
attention output (``o_proj``) and MLP output (``down_proj``). Override
``target_suffixes`` if your model's naming differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kairos.aletheia.surgery.refusal_direction import (
    project_out_subspace,
)


def apply_direction_projection(
    state_dict: dict,
    direction,
    *,
    target_suffixes: list[str] | None = None,
    skip_substrings: list[str] | None = None,
    in_place: bool = False,
) -> tuple[dict, dict]:
    """Standard refusal-direction projection: ``W' = W − r·(rᵀ·W)`` for
    every targeted weight matrix whose row-dim matches ``direction``.

    Factored out of ``CapabilityAwareAbliterator.apply`` so the same
    projection can be applied to either a raw refusal direction (=
    standard Arditi abliteration) or an orthogonalised one (= our
    capability-aware variant).

    Parameters
    ----------
    state_dict : dict[str, Tensor]
    direction : torch.Tensor, shape (d,)
        Unit-norm direction to project out. Must match the row-dim of
        target weight matrices.
    target_suffixes : list[str] | None
        Weight-key suffixes to project. Default:
        ``["o_proj.weight", "down_proj.weight"]``.
    skip_substrings : list[str] | None
        Skip keys containing any substring. Default skips vision /
        audio paths (multimodal models).
    in_place : bool
        Mutate ``state_dict`` directly. Default False (returns a
        shallow copy with new tensor entries for touched keys).

    Returns
    -------
    (new_state_dict, info)
        info = {"touched_layers": list, "n_touched": int, "n_skipped": int}
    """
    import torch

    if target_suffixes is None:
        target_suffixes = ["o_proj.weight", "down_proj.weight"]
    if skip_substrings is None:
        skip_substrings = [
            "vision_tower", "audio_tower", "multi_modal_projector",
            "embed_audio", "embed_vision",
        ]
    r = direction
    if r.dim() != 1:
        raise ValueError(f"direction must be 1-D; got shape {tuple(r.shape)}")

    out = state_dict if in_place else dict(state_dict)
    touched: list[str] = []
    n_skipped = 0
    for name, w in state_dict.items():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        if any(sub in name for sub in skip_substrings):
            n_skipped += 1
            continue
        if w.dim() != 2:
            continue
        if w.shape[0] != r.shape[0]:
            n_skipped += 1
            continue
        # Move r to whatever device/dtype W lives on for the math, then
        # cast result back.
        w_f = w.to(dtype=torch.float32)
        r_dev = r.to(device=w_f.device, dtype=torch.float32)
        # ΔW = r · (rᵀ · W)
        proj = torch.outer(r_dev, r_dev @ w_f)
        out[name] = (w_f - proj).to(w.dtype)
        touched.append(name)
    return out, {
        "touched_layers": touched,
        "n_touched": len(touched),
        "n_skipped": n_skipped,
    }


@dataclass
class AbliterationReport:
    n_touched: int = 0
    n_skipped: int = 0
    refusal_norm_before: float = 0.0
    refusal_norm_after_orthogonalise: float = 0.0
    axis_overlaps: dict[str, float] = field(default_factory=dict)
    touched_layers: list[str] = field(default_factory=list)


class CapabilityAwareAbliterator:
    """Capability-orthogonalised abliteration.

    Parameters
    ----------
    refusal : RefusalDirection
        Output of ``compute_direction_from_activations``.
    capability : CapabilitySubspace
        Output of ``compute_capability_subspace``.
    target_suffixes : list[str] | None
        Weight-key suffixes to apply the projection to. Default:
        ``["o_proj.weight", "down_proj.weight"]``. Add ``"linear.weight"``
        for Gemma 4 (vision / audio wrap their Linear in
        ``Gemma4ClippableLinear``).
    skip_substrings : list[str] | None
        Skip any key containing one of these substrings (defensive
        filter to avoid touching the wrong tower). Default skips vision /
        audio paths.
    """

    def __init__(
        self,
        refusal,
        capability,
        *,
        target_suffixes: list[str] | None = None,
        skip_substrings: list[str] | None = None,
    ) -> None:
        if target_suffixes is None:
            target_suffixes = ["o_proj.weight", "down_proj.weight"]
        if skip_substrings is None:
            skip_substrings = [
                "vision_tower", "audio_tower", "multi_modal_projector",
                "embed_audio", "embed_vision",
            ]
        self.refusal = refusal
        self.capability = capability
        self.target_suffixes = list(target_suffixes)
        self.skip_substrings = list(skip_substrings)
        self.r_pure = None
        self.axis_overlaps: dict[str, float] = {}
        self.report = AbliterationReport()

    def prepare(self) -> "AbliterationReport":
        """Compute ``r_pure`` (refusal-orthogonal-to-capability) and the
        per-axis overlap coefficients. Idempotent — calling this twice
        is harmless.
        """
        r = self.refusal.direction
        # Per-axis overlap: how much of refusal lies along each unit axis
        for axis_name, axis_dir in (self.capability.axis_directions or {}).items():
            denom = float(axis_dir.norm().item())
            if denom <= 1e-12:
                self.axis_overlaps[axis_name] = 0.0
                continue
            overlap = float((r @ (axis_dir / denom)).item())
            self.axis_overlaps[axis_name] = overlap

        before_norm = float(r.norm().item())
        r_pure = project_out_subspace(r, self.capability)
        after_norm = float(r_pure.norm().item())   # 1.0 if non-degenerate

        self.r_pure = r_pure
        self.report.refusal_norm_before = before_norm
        self.report.refusal_norm_after_orthogonalise = after_norm
        self.report.axis_overlaps = dict(self.axis_overlaps)
        return self.report

    def apply(self, state_dict: dict, *, in_place: bool = False) -> dict:
        """Apply capability-aware abliteration to a state dict.

        Returns a new state_dict (or mutates in place if ``in_place=True``)
        with target layers projected by ``r_pure``.
        """
        if self.r_pure is None:
            self.prepare()
        new_sd, info = apply_direction_projection(
            state_dict, self.r_pure,
            target_suffixes=self.target_suffixes,
            skip_substrings=self.skip_substrings,
            in_place=in_place,
        )
        self.report.n_touched = info["n_touched"]
        self.report.n_skipped = info["n_skipped"]
        self.report.touched_layers = info["touched_layers"]
        return new_sd

    # ------------------------------------------------------------------
    # Codebook export
    # ------------------------------------------------------------------

    def export_codebook(self) -> dict:
        """Return the (capability-aware) codebook for this abliteration:
        the orthogonalised refusal direction, the capability subspace,
        the per-axis overlaps, and the list of touched layer names."""
        return {
            "r_pure": self.r_pure,
            "capability_basis": self.capability.basis,
            "capability_axes": list(self.capability.axis_names),
            "axis_overlaps": dict(self.axis_overlaps),
            "touched_layers": list(self.report.touched_layers),
        }
