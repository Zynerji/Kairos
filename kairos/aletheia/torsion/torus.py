"""Torus pendulum T^2 — quasiperiodic weight x schedule modulation.

theta_1 = (2*pi/phi^2) * t ~= 137.5 deg/step  (golden angle)
theta_2 = (2*pi/beta_3) * t ~= 109 deg/step   (bronze angle)

Golden-bronze pair is maximally irrational: the trajectory on T^2
never revisits the same (weight, step-count) combination across
training. Each head gets a golden-angle-spread phase offset.

weight: base * (1 + amp_w * cos(theta_1 + phi_h) * cos(theta_2 + phi_h/phi))
step_count: base_steps * (1 + amp_s * cos(theta_2 + phi_h/phi))

Clamped and rounded; step_count >= 1.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0              # ~= 1.6180
PHI2: float = PHI * PHI                                 # ~= 2.6180
BRONZE_RATIO: float = (3.0 + math.sqrt(13.0)) / 2.0     # ~= 3.3028

THETA1_STEP: float = 2.0 * math.pi / PHI2               # ~= 2.3998 rad (~137.5 deg)
THETA2_STEP: float = 2.0 * math.pi / BRONZE_RATIO       # ~= 1.9022 rad (~109 deg)
GOLDEN_ANGLE: float = math.pi * (3.0 - math.sqrt(5.0))  # ~= 2.3998 rad


@dataclass
class TorusPendulum:
    heads: list[str]
    weight_amplitude: float = 0.4
    step_amplitude: float = 0.3
    floor: float = 0.3
    ceil: float = 2.0
    base_steps: int = 100
    base_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.heads:
            raise ValueError("TorusPendulum requires at least one head")
        if self.floor >= self.ceil:
            raise ValueError(f"floor {self.floor} must be < ceil {self.ceil}")
        if self.base_steps < 1:
            raise ValueError("base_steps must be >= 1")
        self.phases: dict[str, float] = {
            h: (GOLDEN_ANGLE * i) % (2.0 * math.pi)
            for i, h in enumerate(self.heads)
        }
        for h in self.heads:
            self.base_weights.setdefault(h, 1.0)

    def _angles(self, step: int) -> tuple[float, float]:
        return (THETA1_STEP * step, THETA2_STEP * step)

    def weight(self, head: str, step: int) -> float:
        if head not in self.phases:
            raise KeyError(f"unknown head: {head}")
        t1, t2 = self._angles(step)
        ph = self.phases[head]
        mod = 1.0 + self.weight_amplitude * (
            math.cos(t1 + ph) * math.cos(t2 + ph / PHI)
        )
        raw = self.base_weights[head] * mod
        return max(self.floor, min(self.ceil, raw))

    def step_count(self, head: str, step: int) -> int:
        if head not in self.phases:
            raise KeyError(f"unknown head: {head}")
        _, t2 = self._angles(step)
        ph = self.phases[head]
        mod = 1.0 + self.step_amplitude * math.cos(t2 + ph / PHI)
        return max(1, int(round(self.base_steps * mod)))

    def weights(self, step: int) -> dict[str, float]:
        return {h: self.weight(h, step) for h in self.heads}

    def step_counts(self, step: int) -> dict[str, int]:
        return {h: self.step_count(h, step) for h in self.heads}
