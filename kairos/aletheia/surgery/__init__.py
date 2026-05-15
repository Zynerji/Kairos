"""kairos.aletheia.surgery — refusal-direction abliteration and healing.

Two complementary tools:

* ``CapabilityAwareAbliterator`` (Path A) — capability-orthogonalised
  abliteration. Apply at source if you control the abliteration; produces
  abliterated weights with minimal capability collateral. Includes a
  per-axis overlap diagnostic that tells you which capability axes the
  standard recipe would have damaged.

* ``WeightDeltaCodebook`` (Path B) — post-hoc codebook healer for
  already-abliterated checkpoints. Given the un-abliterated + abliterated
  weight pair, build the per-layer ΔW codebook, split each layer's rank-1
  delta into capability-aligned and capability-orthogonal components, and
  selectively re-inject the capability fraction at strength α. Sweep α
  with the Pareto eval loop to pick the healing operating point.

Both use the same primitives in ``refusal_direction``:
``compute_direction_from_activations``, ``compute_capability_subspace``,
``project_out_subspace``.

Example (Path A — capability-aware abliterate)::

    from kairos.aletheia.surgery import (
        compute_direction_from_activations,
        compute_capability_subspace,
        CapabilityAwareAbliterator,
    )

    refusal = compute_direction_from_activations(h_harmful, h_harmless)
    capability = compute_capability_subspace(
        axis_activations={"math": h_math, "facts": h_facts},
        neutral_activations=h_neutral,
    )
    abl = CapabilityAwareAbliterator(refusal, capability)
    abl.prepare()                          # populates axis_overlaps + r_pure
    new_state_dict = abl.apply(state_dict)

Example (Path B — codebook restore)::

    from kairos.aletheia.surgery import (
        compute_capability_subspace,
        WeightDeltaCodebook,
    )

    cap = compute_capability_subspace(axis_acts, h_neutral)
    book = WeightDeltaCodebook()
    book.build(original_state_dict, abliterated_state_dict)
    # Sweep alpha, evaluate per-axis, pick the Pareto-best:
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        healed = book.apply_restoration(abliterated_state_dict, cap, alpha)
        ... load healed, run pool.evaluate(), compare ...
"""

from __future__ import annotations

from kairos.aletheia.surgery.refusal_direction import (
    RefusalDirection,
    CapabilitySubspace,
    compute_direction_from_activations,
    compute_capability_subspace,
    project_out_subspace,
)
from kairos.aletheia.surgery.codebook import (
    LayerDelta,
    CodebookReport,
    WeightDeltaCodebook,
)
from kairos.aletheia.surgery.capability_aware_abliterate import (
    AbliterationReport,
    CapabilityAwareAbliterator,
)

__all__ = [
    "AbliterationReport",
    "CapabilityAwareAbliterator",
    "CapabilitySubspace",
    "CodebookReport",
    "LayerDelta",
    "RefusalDirection",
    "WeightDeltaCodebook",
    "compute_capability_subspace",
    "compute_direction_from_activations",
    "project_out_subspace",
]
