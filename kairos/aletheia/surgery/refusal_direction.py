"""Compute refusal direction and capability subspace from residual-stream
activations.

The refusal-direction technique (Arditi et al., "Refusal in Language Models
Is Mediated by a Single Direction", 2024): take the difference of means
between residual-stream activations on harmful prompts and harmless prompts.
That difference, normalised, is the direction the model uses to mediate
refusal. Project it out of every weight matrix that writes to the residual
stream and refusals go away.

Same diff-of-means trick gives you an orthonormal basis for any other axis
you care about (capability axes: math, factuality, reasoning, instruction
following, etc.) — collect prompts that exercise the capability vs neutral
prompts, take the diff, gather across axes, QR-orthogonalise.

We use these two primitives downstream:

* In ``CapabilityAwareAbliterator`` (Path A — inline at abliteration time)
  to orthogonalise the refusal direction against the capability subspace
  BEFORE projecting, so the projection only removes refusal-orthogonal-to-
  capability content.

* In ``WeightDeltaCodebook`` (Path B — post-hoc on existing abliterated
  models) to classify each rank-1 component of W_original − W_abliterated
  as refusal vs capability content, so we can selectively re-inject the
  capability fraction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefusalDirection:
    """A normalised refusal direction vector + the per-prompt-class means
    it was computed from."""
    direction: object              # torch.Tensor, shape (d,), unit norm
    harmful_mean: object           # torch.Tensor, shape (d,)
    harmless_mean: object          # torch.Tensor, shape (d,)
    n_harmful: int
    n_harmless: int


@dataclass
class CapabilitySubspace:
    """An orthonormal basis spanning K capability axes."""
    basis: object                  # torch.Tensor, shape (d, k) — columns are unit-norm + orthogonal
    axis_names: list[str]
    axis_directions: dict[str, object]   # per-axis pre-orthogonalisation raw direction (d,)


def compute_direction_from_activations(
    harmful_activations,
    harmless_activations,
    *, normalise: bool = True,
) -> "RefusalDirection":
    """Diff-of-means on a single residual-stream slice.

    Parameters
    ----------
    harmful_activations : torch.Tensor, shape (N_h, d) or (N_h, T_h, d)
        Residual-stream activations on harmful prompts. If 3D, we mean
        across the time axis first (pool over tokens).
    harmless_activations : torch.Tensor, shape (N_b, d) or (N_b, T_b, d)
    normalise : bool
        If True, return a unit-norm direction. Default True.
    """
    import torch

    h_hf = harmful_activations
    h_bn = harmless_activations
    if h_hf.dim() == 3:
        h_hf = h_hf.mean(dim=1)
    if h_bn.dim() == 3:
        h_bn = h_bn.mean(dim=1)
    if h_hf.dim() != 2 or h_bn.dim() != 2:
        raise ValueError(
            f"activations must be 2D or 3D; got harmful {tuple(h_hf.shape)}, "
            f"harmless {tuple(h_bn.shape)}"
        )
    if h_hf.shape[-1] != h_bn.shape[-1]:
        raise ValueError(
            f"hidden-dim mismatch: harmful {h_hf.shape[-1]} vs "
            f"harmless {h_bn.shape[-1]}"
        )

    harmful_mean = h_hf.float().mean(dim=0)
    harmless_mean = h_bn.float().mean(dim=0)
    r = harmful_mean - harmless_mean
    if normalise:
        norm = r.norm()
        if float(norm.item()) > 1e-12:
            r = r / norm
    return RefusalDirection(
        direction=r,
        harmful_mean=harmful_mean,
        harmless_mean=harmless_mean,
        n_harmful=int(h_hf.shape[0]),
        n_harmless=int(h_bn.shape[0]),
    )


def compute_capability_subspace(
    axis_activations: dict,
    neutral_activations,
    *, max_rank: int | None = None,
) -> "CapabilitySubspace":
    """Build an orthonormal basis spanning K capability axes.

    Parameters
    ----------
    axis_activations : dict[str, torch.Tensor]
        Maps capability-axis name -> residual-stream activations on
        prompts exercising that capability. Each tensor is (N_i, d) or
        (N_i, T_i, d).
    neutral_activations : torch.Tensor, shape (N_n, d) or (N_n, T_n, d)
        Neutral-prompt activations (the baseline to diff against).
    max_rank : int | None
        Optional cap on the basis rank; useful when you have many axes
        and want to keep the projection cheap. Default: no cap.
    """
    import torch

    if not axis_activations:
        raise ValueError("axis_activations must contain >=1 axis")

    h_neut = neutral_activations
    if h_neut.dim() == 3:
        h_neut = h_neut.mean(dim=1)
    neutral_mean = h_neut.float().mean(dim=0)

    raw_dirs: dict[str, object] = {}
    cols: list[object] = []
    names: list[str] = []
    for name, acts in axis_activations.items():
        h = acts
        if h.dim() == 3:
            h = h.mean(dim=1)
        if h.shape[-1] != neutral_mean.shape[-1]:
            raise ValueError(
                f"axis {name!r} hidden dim {h.shape[-1]} != neutral "
                f"{neutral_mean.shape[-1]}"
            )
        c = h.float().mean(dim=0) - neutral_mean
        raw_dirs[name] = c
        n = c.norm()
        if float(n.item()) <= 1e-12:
            # Skip degenerate axes
            continue
        cols.append(c / n)
        names.append(name)

    if not cols:
        raise ValueError("all capability axes were degenerate (zero diff-of-means)")

    C_raw = torch.stack(cols, dim=1)        # (d, k_raw)
    Q, _ = torch.linalg.qr(C_raw)            # (d, k_raw), orthonormal columns
    k_actual = Q.shape[1]
    if max_rank is not None and k_actual > max_rank:
        Q = Q[:, :max_rank]
        names = names[:max_rank]

    return CapabilitySubspace(
        basis=Q, axis_names=names, axis_directions=raw_dirs,
    )


def project_out_subspace(direction, subspace) -> "torch.Tensor":
    """Return ``direction`` with its projection onto ``subspace.basis``
    removed (Gram-Schmidt). Renormalised to unit length unless the
    resulting vector is too small (in which case raw zeros are returned).

    This is the core of Path A: ``r_pure = r − C·(Cᵀ·r)``.
    """
    import torch

    r = direction
    if hasattr(subspace, "basis"):
        C = subspace.basis
    else:
        C = subspace
    if C.dim() != 2 or r.dim() != 1:
        raise ValueError(
            f"expected C (d,k) and r (d,); got C {tuple(C.shape)} r {tuple(r.shape)}"
        )
    if C.shape[0] != r.shape[0]:
        raise ValueError(
            f"hidden-dim mismatch: r {r.shape[0]} vs C {C.shape[0]}"
        )
    coeff = C.t() @ r                # (k,)
    r_pure = r - C @ coeff
    norm = r_pure.norm()
    if float(norm.item()) > 1e-12:
        r_pure = r_pure / norm
    return r_pure
