"""KairosParetoGuard — multi-axis Pareto-frontier rollback gate.

Ported from Aletheia's `src/aletheia/ratchet/pareto.py` + spectral
amplifier. Proven in jDHART post-training on Qwen3 (9 Pareto axes:
factuality, calibration, abstention, grounding, consistency,
sycophancy, reasoning, instruction, distillation) with dual-regression
rollback gate.

Mechanism (Pareto ratchet):
  * `anchor`: per-axis baseline scores
  * `product`: P(scores) = prod_i max(scores[i], eps)
  * `floor`: per-axis lower bound = `floor_mult * anchor[i]`  (default 0.80)
  * `new best`: product improves AND no axis is below floor
  * `rollback`: ≥ 2 axes below floor simultaneously (dual-regression gate)

Aletheia found a single-axis dip is expected in torsion-cycling phases
(some metrics naturally regress when one pool is actively trained).
Triggering rollback on single-axis dips would oscillate; requiring two
simultaneous dips gives stability while still catching real regressions.

Mechanism (spectral amplification):
  Optional companion: per-head `alpha = -log(head_std / target_std)`,
  clamped to [0, alpha_max]. When a head's variance collapses, alpha
  rises and signals the training loop to up-weight extreme-target
  samples (see `aletheia.torsion.spectral_amp`). Kairos exposes the
  computed alpha in the Action notes; the training loop applies it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from grokking_monitor import GrokkingMonitor

from .core import Action, BaseCallback


@dataclass(frozen=True)
class ParetoState:
    """Snapshot of the current Pareto state."""
    best_product: float
    best_scores: dict
    last_product: float
    last_scores: dict
    last_below_floor_axes: list
    last_is_new_best: bool
    last_should_rollback: bool


class KairosParetoGuard(BaseCallback):
    """Pareto-frontier rollback + new-best detector for multi-axis metrics.

    Parameters
    ----------
    anchor : dict[str, float]
        Per-axis baseline scores. Each axis is treated as a Pareto
        coordinate. Required.
    floor_mult : float
        Per-axis floor = ``floor_mult * anchor[axis]``. Default 0.80
        (proven on jDHART Phase B).
    eps : float
        Numerical floor for the product metric (avoids log/zero).
    metric_prefix : str
        Only metrics with keys starting with this prefix are read as
        Pareto axes. Default ``"pool_"`` (matches Aletheia's pool naming).
        Set to "" to read every keyword passed in.
    spectral_target_std : float | None
        If provided, optionally compute a per-axis spectral
        amplification alpha relative to this target std. Useful when
        you also pass per-axis std values as ``f"{axis}_std"``.
    alpha_max : float
        Clamp on the spectral alpha. Default 5.0 (Aletheia value).
    """

    name = "KairosParetoGuard"

    def __init__(self, anchor: dict[str, float], floor_mult: float = 0.80,
                 eps: float = 1e-3, metric_prefix: str = "pool_",
                 spectral_target_std: float | None = None,
                 alpha_max: float = 5.0) -> None:
        if not anchor:
            raise ValueError("anchor must have at least one axis")
        if not (0.0 < floor_mult <= 1.0):
            raise ValueError(f"floor_mult must be in (0,1]; got {floor_mult}")
        self.anchor = {str(k): float(v) for k, v in anchor.items()}
        self.floor_mult = float(floor_mult)
        self.eps = float(eps)
        self.metric_prefix = str(metric_prefix)
        self.spectral_target_std = (
            float(spectral_target_std) if spectral_target_std is not None else None
        )
        self.alpha_max = float(alpha_max)
        self.best_product: float = self._product(self.anchor)
        self.best_scores: dict[str, float] = dict(self.anchor)
        self._n_rollback_signals: int = 0
        self._n_new_best_signals: int = 0
        self._last_state: ParetoState | None = None

    @property
    def n_rollback_signals(self) -> int:
        return self._n_rollback_signals

    @property
    def n_new_best_signals(self) -> int:
        return self._n_new_best_signals

    @property
    def last_state(self) -> ParetoState | None:
        return self._last_state

    # ------------------------------------------------------------------
    # Core helpers (mirroring aletheia.ratchet.ParetoRatchet)
    # ------------------------------------------------------------------

    def _product(self, scores: dict[str, float]) -> float:
        p = 1.0
        for k in self.anchor:
            p *= max(float(scores.get(k, 0.0)), self.eps)
        return p

    def _below_floor(self, scores: dict[str, float]) -> list[str]:
        return [
            k for k, anc in self.anchor.items()
            if float(scores.get(k, 0.0)) < self.floor_mult * anc
        ]

    def _is_new_best(self, scores: dict[str, float]) -> bool:
        return (
            self._product(scores) > self.best_product
            and not self._below_floor(scores)
        )

    def _should_rollback(self, scores: dict[str, float]) -> bool:
        return len(self._below_floor(scores)) >= 2

    @staticmethod
    def _adaptive_alpha(head_std: float, target_std: float,
                         alpha_max: float, eps: float = 1e-8) -> float:
        if target_std < eps:
            return 0.0
        if head_std < eps:
            return float(alpha_max)
        ratio = head_std / target_std
        return max(0.0, min(alpha_max, -math.log(ratio)))

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        # Gather per-axis Pareto scores from kwargs
        scores: dict[str, float] = {}
        for k in self.anchor:
            full_key = f"{self.metric_prefix}{k}"
            if full_key in metrics and isinstance(metrics[full_key], (int, float)):
                scores[k] = float(metrics[full_key])
            elif k in metrics and isinstance(metrics[k], (int, float)):
                scores[k] = float(metrics[k])
        if not scores:
            return Action()

        product = self._product(scores)
        below = self._below_floor(scores)
        is_new_best = self._is_new_best(scores)
        rollback = self._should_rollback(scores)

        notes: list[str] = []
        save_ckpt = False
        ckpt_tag = ""

        if is_new_best:
            self.best_product = product
            self.best_scores = dict(scores)
            save_ckpt = True
            ckpt_tag = f"pareto_best_step{step}"
            self._n_new_best_signals += 1
            notes.append(
                f"pareto: NEW BEST  product={product:.4f}  "
                f"prev_best={self.best_product:.4f}"
            )
        if rollback:
            self._n_rollback_signals += 1
            notes.append(
                f"pareto: ROLLBACK signal  below_floor={below}  "
                f"product={product:.4f}"
            )

        # Optional spectral-amplification companion. Reads "<axis>_std".
        if self.spectral_target_std is not None:
            for k in self.anchor:
                std_key = f"{self.metric_prefix}{k}_std"
                if std_key in metrics and isinstance(metrics[std_key], (int, float)):
                    alpha = self._adaptive_alpha(
                        float(metrics[std_key]),
                        self.spectral_target_std,
                        self.alpha_max,
                    )
                    if alpha > 0.5:
                        notes.append(
                            f"spectral_amp: axis={k} alpha={alpha:.3f} "
                            f"(head_std={metrics[std_key]:.4f})"
                        )

        self._last_state = ParetoState(
            best_product=self.best_product,
            best_scores=dict(self.best_scores),
            last_product=product,
            last_scores=dict(scores),
            last_below_floor_axes=list(below),
            last_is_new_best=bool(is_new_best),
            last_should_rollback=bool(rollback),
        )

        return Action(
            save_checkpoint=save_ckpt,
            checkpoint_tag=ckpt_tag,
            stop_training=False,  # rollback is signal, not abort
            notes=notes,
        )
