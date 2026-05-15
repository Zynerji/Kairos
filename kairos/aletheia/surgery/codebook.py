"""WeightDeltaCodebook — Path B post-hoc codebook recovery for already-
abliterated checkpoints.

You have two checkpoints:

* ``W_original`` — the un-abliterated base model
* ``W_abliterated`` — the abliterated derivative (e.g. huihui-ai, OBLITERATUS)

Standard abliteration applies a rank-1 projection per touched layer:

    W_abliterated_ℓ = W_original_ℓ − r·rᵀ·W_original_ℓ / ‖r‖²

So the layer-wise weight delta IS the codebook:

    ΔW_ℓ = W_original_ℓ − W_abliterated_ℓ            (≈ rank-1)

This module:

1. Pair-matches layer names between the two state dicts (suffix-based,
   handles HF / PEFT-style naming).
2. Computes ``ΔW_ℓ`` per matched pair and SVD-decomposes.
3. Given a ``CapabilitySubspace`` (from
   ``kairos.aletheia.surgery.refusal_direction.compute_capability_subspace``)
   splits each rank-1 component into refusal-orthogonal-to-capability and
   refusal-co-aligned-with-capability fractions.
4. Selective re-injection: produce a healed state_dict where ``α``
   fraction of the capability fraction is added back, leaving the
   refusal-orthogonal fraction removed.

The re-injection scalar ``α`` is sweep-able (the Pareto eval loop is the
natural validator: pick the ``α`` that maximises per-axis task scores
without re-introducing refusal).

Mathematical identity
=====================

For a rank-1 ΔW = u·sᵀ·vᵀ where u, v are unit vectors and s is the
singular value:

    u = u_capability + u_refusal                    where
        u_capability = C·(Cᵀ·u),
        u_refusal    = u − u_capability             (orthogonal to capability subspace)

    ΔW = ΔW_capability + ΔW_refusal
        = u_capability·s·vᵀ + u_refusal·s·vᵀ

W_healed(α) = W_abliterated + α · ΔW_capability     (α ∈ [0, 1])
            = W_abliterated + α · C·Cᵀ·ΔW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayerDelta:
    """Per-layer codebook entry."""
    name: str
    shape: tuple
    u: Any                          # torch.Tensor, (d_out,), unit norm
    s: Any                          # torch.Tensor, scalar (top singular value)
    v: Any                          # torch.Tensor, (d_in,), unit norm
    rank1_fraction: float           # s_1^2 / sum(s^2), how rank-1 the delta is
    full_delta: Any | None = None   # optional full ΔW for non-rank-1 deltas


@dataclass
class CodebookReport:
    n_paired: int = 0
    n_skipped: int = 0
    layer_deltas: list[LayerDelta] = field(default_factory=list)


class WeightDeltaCodebook:
    """A/B codebook builder + selective restorer.

    Parameters
    ----------
    keep_full_delta : bool
        If True, keep the full ΔW per layer (memory-heavy). If False
        (default), keep only the rank-1 SVD compression. Set True if
        the abliteration was rank-k for k>1.
    rank1_threshold : float
        Layers whose top singular value captures less than this fraction
        of total spectrum energy are flagged (likely not standard
        rank-1 abliteration). Default 0.95.
    """

    def __init__(self, *, keep_full_delta: bool = False,
                 rank1_threshold: float = 0.95) -> None:
        if not (0.0 < rank1_threshold <= 1.0):
            raise ValueError("rank1_threshold must be in (0, 1]")
        self.keep_full_delta = bool(keep_full_delta)
        self.rank1_threshold = float(rank1_threshold)
        self._layers: dict[str, LayerDelta] = {}
        self.report = CodebookReport()

    # ------------------------------------------------------------------
    # Building the codebook
    # ------------------------------------------------------------------

    def build(self, original_state_dict: dict, abliterated_state_dict: dict,
              *, target_suffixes: list[str] | None = None) -> "CodebookReport":
        """Build the per-layer codebook from a pair of state dicts.

        Parameters
        ----------
        original_state_dict, abliterated_state_dict : dict
            Both should have the same module-tree naming. We pair on
            full key equality and only diff layers whose name ends in
            one of ``target_suffixes`` (default: ``[".weight"]`` for
            layers that abliteration typically touches — o_proj,
            down_proj, etc.).
        """
        import torch

        if target_suffixes is None:
            target_suffixes = [".weight"]

        self._layers.clear()
        self.report = CodebookReport()

        for name, w_orig in original_state_dict.items():
            if not any(name.endswith(s) for s in target_suffixes):
                continue
            if name not in abliterated_state_dict:
                self.report.n_skipped += 1
                continue
            w_abl = abliterated_state_dict[name]
            if w_abl.shape != w_orig.shape:
                self.report.n_skipped += 1
                continue
            if w_orig.dim() != 2:
                # We only abliterate 2D projection weights; skip biases,
                # norms, embeddings.
                continue

            delta = (w_orig.float() - w_abl.float())
            # Quick check: layer untouched by abliteration → delta ~ 0
            if float(delta.norm().item()) < 1e-8:
                continue

            # Top-1 SVD
            try:
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            except RuntimeError:
                self.report.n_skipped += 1
                continue
            total_energy = float((S * S).sum().item())
            top_energy = float((S[0] * S[0]).item())
            r1_frac = top_energy / max(total_energy, 1e-12)

            entry = LayerDelta(
                name=name,
                shape=tuple(delta.shape),
                u=U[:, 0].clone(),
                s=S[0].clone(),
                v=Vh[0, :].clone(),
                rank1_fraction=r1_frac,
                full_delta=delta.clone() if self.keep_full_delta else None,
            )
            self._layers[name] = entry
            self.report.layer_deltas.append(entry)
            self.report.n_paired += 1
        return self.report

    # ------------------------------------------------------------------
    # Inspecting
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._layers)

    def layers(self) -> list[str]:
        return list(self._layers.keys())

    def get(self, name: str) -> LayerDelta:
        return self._layers[name]

    def non_rank1_layers(self) -> list[str]:
        return [n for n, e in self._layers.items()
                if e.rank1_fraction < self.rank1_threshold]

    # ------------------------------------------------------------------
    # Decomposing against a capability subspace
    # ------------------------------------------------------------------

    def split_against_capability(self, subspace) -> dict:
        """Split each layer's rank-1 ΔW into capability-aligned and
        capability-orthogonal components.

        Returns a dict ``{layer_name: {"u_cap": ..., "u_ref": ...,
        "alpha_cap": ...}}`` where ``alpha_cap`` is the fraction of
        the top-1 direction that lies in the capability subspace
        (`‖C·Cᵀ·u‖`). Layers with alpha_cap ≈ 0 are essentially
        pure refusal; layers with alpha_cap close to 1 are mostly
        capability and abliteration hurt them.
        """
        import torch

        C = subspace.basis if hasattr(subspace, "basis") else subspace
        out: dict[str, dict] = {}
        for name, entry in self._layers.items():
            u = entry.u
            if u.shape[0] != C.shape[0]:
                continue
            coeff = C.t() @ u                 # (k,)
            u_cap = C @ coeff                  # (d_out,) capability-aligned component
            u_ref = u - u_cap                  # (d_out,) capability-orthogonal component
            alpha_cap = float(u_cap.norm().item())   # in [0, 1] since u is unit
            out[name] = dict(
                u_cap=u_cap, u_ref=u_ref, alpha_cap=alpha_cap,
                s=entry.s, v=entry.v,
            )
        return out

    # ------------------------------------------------------------------
    # Re-injection
    # ------------------------------------------------------------------

    def apply_restoration(self, abliterated_state_dict: dict,
                           subspace,
                           alpha: float = 1.0,
                           *, in_place: bool = False) -> dict:
        """Produce a healed state_dict where the capability-aligned
        fraction of each ΔW is added back at strength α.

        ``alpha`` = 1.0 → fully restore the capability projection.
        ``alpha`` = 0.0 → return abliterated weights unchanged.
        ``alpha`` between → partial restore (sweep over α to find the
        Pareto-optimal point).
        """
        import torch

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        out: dict = abliterated_state_dict if in_place else dict(abliterated_state_dict)
        splits = self.split_against_capability(subspace)
        for name, parts in splits.items():
            u_cap = parts["u_cap"]
            s = parts["s"]
            v = parts["v"]
            # ΔW_cap_layer = u_cap · s · vᵀ
            delta_cap = torch.outer(u_cap, v) * float(s.item())
            w = out[name].float()
            out[name] = (w + alpha * delta_cap).to(out[name].dtype)
        return out
