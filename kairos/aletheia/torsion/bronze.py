"""Bronze pendulum — anti-resonant per-head loss weighting.

Bronze ratio beta_3 = (3 + sqrt(13)) / 2, solves x^2 = 3x + 1.
Bronze angle = 2*pi/beta_3 ~= 1.9022 rad ~= 109 deg per step.

Maximally irrational with respect to golden phi-structured backbones.
Prevents loss-weight harmonics from settling on any single head's
optimum during Phase A training.

Each head gets a fixed phase offset spread by the golden angle.

weight(head, step) = base[head] * (1 + amp * cos(bronze_angle*step + phase[head]))
clamped to [floor, ceil].
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

BRONZE_RATIO: float = (3.0 + math.sqrt(13.0)) / 2.0          # ~= 3.3028
BRONZE_ANGLE: float = 2.0 * math.pi / BRONZE_RATIO           # ~= 1.9022 rad (~109 deg)
GOLDEN_ANGLE: float = math.pi * (3.0 - math.sqrt(5.0))       # ~= 2.3998 rad (~137.5 deg)


@dataclass
class BronzePendulum:
    heads: list[str]
    amplitude: float = 0.4
    floor: float = 0.3
    ceil: float = 2.0
    base_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.heads:
            raise ValueError("BronzePendulum requires at least one head")
        if self.floor >= self.ceil:
            raise ValueError(f"floor {self.floor} must be < ceil {self.ceil}")
        self.phases: dict[str, float] = {
            h: (GOLDEN_ANGLE * i) % (2.0 * math.pi)
            for i, h in enumerate(self.heads)
        }
        for h in self.heads:
            self.base_weights.setdefault(h, 1.0)

    def weight(self, head: str, step: int) -> float:
        if head not in self.phases:
            raise KeyError(f"unknown head: {head}")
        phase = self.phases[head]
        mod = 1.0 + self.amplitude * math.cos(BRONZE_ANGLE * step + phase)
        raw = self.base_weights[head] * mod
        return max(self.floor, min(self.ceil, raw))

    def weights(self, step: int) -> dict[str, float]:
        return {h: self.weight(h, step) for h in self.heads}
