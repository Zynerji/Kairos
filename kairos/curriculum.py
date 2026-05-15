"""KairosCurriculum: phase-aware optimizer settings.

Defines what (lr, weight_decay, accelerator_active) the optimizer
SHOULD have in each Phase. The CallbackBundle's `current_phase`
determines which `PhaseSettings` applies. The callback emits the
required lr / weight_decay multipliers in its Action.

The default settings are:
  * MEMORISING:  lr 1.0x, wd 1.0x, accelerator off
  * PLATEAU:     lr 1.0x, wd 1.0x, accelerator on (research)
  * NEAR_CRITICAL: lr 0.1x (drop), wd 1.0x, accelerator off
  * DRIFTING:    lr 0.1x (held), wd 1.0x, accelerator off
  * POST:        lr 0.1x (held), wd 0.5x (reduce; we want to keep
                 the learned representation), accelerator off

These are heuristic defaults; override `settings` in the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grokking_monitor import GrokkingMonitor

from .core import Action, BaseCallback, Phase


@dataclass(frozen=True)
class PhaseSettings:
    """Optimizer-side adjustments for one Phase, expressed as
    multiplicative factors over the initial / current values."""
    lr_multiplier: float = 1.0
    weight_decay_multiplier: float = 1.0
    accelerator_active: bool = False


_DEFAULT_SETTINGS: dict[Phase, PhaseSettings] = {
    Phase.MEMORISING:   PhaseSettings(1.0, 1.0, False),
    Phase.PLATEAU:      PhaseSettings(1.0, 1.0, True),
    Phase.NEAR_CRITICAL: PhaseSettings(0.1, 1.0, False),
    Phase.DRIFTING:     PhaseSettings(0.1, 1.0, False),
    Phase.POST:         PhaseSettings(0.1, 0.5, False),
    Phase.UNKNOWN:      PhaseSettings(1.0, 1.0, False),
}


class KairosCurriculum(BaseCallback):
    """Phase-aware optimizer-settings dispatcher.

    Parameters
    ----------
    settings : dict[Phase, PhaseSettings] | None
        Mapping of phase to desired settings. Missing entries fall
        back to defaults.
    optimizer : torch.optim.Optimizer | None
        If provided, mutates ``param_groups[*]["lr"]`` and
        ``param_groups[*]["weight_decay"]`` directly when the phase
        changes. Stores the *initial* per-group values on first call
        so settings are applied multiplicatively from the baseline.
    """

    name = "KairosCurriculum"

    def __init__(self, settings: dict[Phase, PhaseSettings] | None = None,
                 optimizer: Any | None = None) -> None:
        merged = dict(_DEFAULT_SETTINGS)
        if settings is not None:
            for k, v in settings.items():
                merged[Phase(k) if isinstance(k, str) else k] = v
        self.settings = merged
        self.optimizer = optimizer
        self._last_phase: Phase = Phase.UNKNOWN
        self._initial: list[tuple[float, float]] | None = None

    def _capture_initial(self) -> None:
        if self.optimizer is None or self._initial is not None:
            return
        self._initial = [
            (float(g["lr"]), float(g.get("weight_decay", 0.0)))
            for g in self.optimizer.param_groups
        ]

    def _apply_to_optimizer(self, settings: PhaseSettings) -> None:
        if self.optimizer is None or self._initial is None:
            return
        for g, (lr0, wd0) in zip(self.optimizer.param_groups, self._initial):
            g["lr"] = lr0 * settings.lr_multiplier
            g["weight_decay"] = wd0 * settings.weight_decay_multiplier

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        self._capture_initial()
        # Use the bundle-supplied phase if available; else infer from
        # train_acc/test_acc fallbacks the same way core._classify_phase
        # does (but we usually get it via Action.phase merging).
        from .core import _classify_phase
        phase = _classify_phase(
            monitor, metrics.get("train_acc"), metrics.get("test_acc"),
        )
        settings = self.settings.get(phase, _DEFAULT_SETTINGS[Phase.UNKNOWN])
        if phase != self._last_phase:
            self._apply_to_optimizer(settings)
            self._last_phase = phase
            return Action(
                lr_multiplier=settings.lr_multiplier,
                weight_decay_multiplier=settings.weight_decay_multiplier,
                phase=phase,
                notes=[
                    f"curriculum: entered {phase.value} -> "
                    f"lr_mul={settings.lr_multiplier}, "
                    f"wd_mul={settings.weight_decay_multiplier}"
                ],
            )
        return Action(phase=phase)
