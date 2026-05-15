"""MoE expert addition stub.

GROWTH MODULE — not active in core fine-tuning. Enable via
configs/growth.yaml -> expert_addition.enabled = true.

Adds N new routed experts to an existing MoE layer:
1. Copy a "template" expert (average or specific one) as initialization
2. Extend router's output projection rows to accommodate new experts
3. Apply small random perturbation to break symmetry with template
4. Re-normalize gate softmax constants

Param cost: ~(H + 3 * H * I_expert) per expert, ~250M in Qwen3-42B-A3B.

NOT IMPLEMENTED — this is a surgical operation that requires model-
specific knowledge. Implement when ready to grow beyond 42B.

For jDHART parallel: similar in spirit to adding a new head to
MultiPoolModel, but the router makes it substantially harder.
"""
from __future__ import annotations
from typing import Any


def add_experts_to_moe(
    moe_module: Any,
    n_new_experts: int,
    template_expert_idx: int | None = None,
    perturbation_std: float = 0.02,
) -> None:
    """Add n_new_experts routed experts to an MoE module (in-place).

    Args:
        moe_module: the MoE layer to modify (architecture-specific)
        n_new_experts: how many new routed experts to add (>= 1)
        template_expert_idx: which existing expert to clone as init;
            None = average of all existing experts
        perturbation_std: gaussian noise std applied to clones to break
            symmetry (0.02 is a conservative default)

    Raises:
        NotImplementedError: always, until wired for specific MoE impl.
    """
    raise NotImplementedError(
        "Expert addition reserved for future. "
        "Enable in configs/growth.yaml when ready to grow beyond 42B. "
        "Implementation requires: (1) expert cloning, (2) router "
        "projection extension, (3) symmetry-breaking perturbation, "
        "(4) gate re-normalization. See docstring."
    )
