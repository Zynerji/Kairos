"""Pareto ratchet for multi-metric torsion training.

Protects the multi-axis frontier across pool rewards.

- Product metric: P(scores) = prod_i max(scores[i], eps)
- Floor check: axis i is "below floor" if scores[i] < floor * anchor[i]
- New best: product improves AND no axis is below floor
- Rollback: >= 2 axes below floor simultaneously (dual-regression)

Dual-regression rollback tolerates Phase B oscillation (single-axis
dips are expected; two-axis collapse is not). 80% floor; not 85%.

Protects e.g.:
    factuality up without abstention down
    safety up without helpfulness down
    reasoning up without instruction-following down
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParetoRatchet:
    anchor: dict[str, float]
    floor: float = 0.80
    eps: float = 1e-3
    best_product: float = field(init=False)
    best_scores: dict[str, float] = field(init=False)
    best_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        if not self.anchor:
            raise ValueError("anchor must have at least one axis")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError(f"floor must be in (0, 1], got {self.floor}")
        self.best_scores = dict(self.anchor)
        self.best_product = self.product(self.anchor)

    def product(self, scores: dict[str, float]) -> float:
        p = 1.0
        for k in self.anchor:
            p *= max(scores.get(k, 0.0), self.eps)
        return p

    def below_floor_axes(self, scores: dict[str, float]) -> list[str]:
        return [
            k for k, v in self.anchor.items()
            if scores.get(k, 0.0) < self.floor * v
        ]

    def should_rollback(self, scores: dict[str, float]) -> bool:
        """Rollback iff >= 2 axes below floor simultaneously."""
        return len(self.below_floor_axes(scores)) >= 2

    def is_new_best(self, scores: dict[str, float]) -> bool:
        return (
            self.product(scores) > self.best_product
            and len(self.below_floor_axes(scores)) == 0
        )

    def update(self, scores: dict[str, float], checkpoint: Path | None = None) -> None:
        self.best_product = self.product(scores)
        self.best_scores = dict(scores)
        if checkpoint is not None:
            self.best_checkpoint = checkpoint
