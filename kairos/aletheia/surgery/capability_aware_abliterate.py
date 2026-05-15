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
        import torch

        if self.r_pure is None:
            self.prepare()
        r = self.r_pure

        out = state_dict if in_place else dict(state_dict)
        self.report.n_touched = 0
        self.report.n_skipped = 0
        self.report.touched_layers = []

        for name, w in state_dict.items():
            if not any(name.endswith(s) for s in self.target_suffixes):
                continue
            if any(sub in name for sub in self.skip_substrings):
                self.report.n_skipped += 1
                continue
            if w.dim() != 2:
                continue
            # W writes from input space (d_in) into residual stream (d_out).
            # r lives in d_out. The output along r is r · (rᵀ · W).
            if w.shape[0] != r.shape[0]:
                # Hidden dim mismatch — refusal direction was computed at
                # a different layer width than this one. Skip silently.
                self.report.n_skipped += 1
                continue
            # ΔW = r · (rᵀ · W)
            w_f = w.float()
            proj = torch.outer(r, r @ w_f)      # (d_out, d_in)
            out[name] = (w_f - proj).to(w.dtype)
            self.report.n_touched += 1
            self.report.touched_layers.append(name)
        return out

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
