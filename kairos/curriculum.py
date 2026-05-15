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
                 optimizer: Any | None = None,
                 hysteresis_steps: int = 800,
                 ratchet_phases: tuple[Phase, ...] = (
                     Phase.NEAR_CRITICAL, Phase.DRIFTING, Phase.POST,
                 )) -> None:
        """Hysteresis-stabilised phase-aware optimizer settings.

        On the slow-grok regime Cassandra's regime flips between
        `drifting` and `stable` every ~200 steps as the test_loss
        noise lands above/below threshold within the diagnostic
        window. The unstabilised curriculum responded by flipping
        LR/WD just as fast, which broke training.

        Two stabilisations:

          - `hysteresis_steps`: once a phase changes, hold it for
            at least this many steps before allowing another change.
            Filters Cassandra's per-check noise.
          - `ratchet_phases`: once we enter one of these "deeper"
            phases (the transition is happening), do NOT go back to
            an earlier phase. Phases ratchet forward through the
            training run.
        """
        merged = dict(_DEFAULT_SETTINGS)
        if settings is not None:
            for k, v in settings.items():
                merged[Phase(k) if isinstance(k, str) else k] = v
        self.settings = merged
        self.optimizer = optimizer
        self.hysteresis_steps = int(hysteresis_steps)
        self.ratchet_phases = tuple(ratchet_phases)
        self._last_phase: Phase = Phase.UNKNOWN
        self._last_change_step: int = -1
        self._initial: list[tuple[float, float]] | None = None
        # Ratchet ordering: higher index = "deeper" in the training
        # progression. Once we reach a ratchet phase at this index or
        # higher, never regress below it.
        self._phase_order: dict[Phase, int] = {
            Phase.UNKNOWN: 0,
            Phase.MEMORISING: 1,
            Phase.PLATEAU: 2,
            Phase.NEAR_CRITICAL: 3,
            Phase.DRIFTING: 3,           # equal weight to near_critical
            Phase.POST: 4,
        }
        self._max_seen_order: int = 0

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

    def _ratchet_filter(self, proposed: Phase) -> Phase:
        """Don't allow regressing below the deepest phase we've seen.

        Once we've been in `near_critical` or `drifting`, we stay
        there (or move to `post`); we never go back to `plateau` /
        `memorising`. Filters Cassandra's per-check noise.
        """
        ord_new = self._phase_order.get(proposed, 0)
        self._max_seen_order = max(self._max_seen_order, ord_new)
        if ord_new < self._max_seen_order:
            # We've seen a deeper phase already; refuse to regress.
            for ph, o in self._phase_order.items():
                if o == self._max_seen_order:
                    return ph
        return proposed

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        self._capture_initial()
        from .core import _classify_phase
        raw_phase = _classify_phase(
            monitor, metrics.get("train_acc"), metrics.get("test_acc"),
        )
        # Apply ratchet: don't regress below the deepest phase seen
        phase = self._ratchet_filter(raw_phase)
        # Apply hysteresis: don't flip phase faster than the cooldown
        if phase != self._last_phase:
            if (self._last_change_step >= 0
                    and step - self._last_change_step < self.hysteresis_steps):
                # In cooldown; ignore the proposed change
                return Action(phase=self._last_phase)
            settings = self.settings.get(phase, _DEFAULT_SETTINGS[Phase.UNKNOWN])
            self._apply_to_optimizer(settings)
            self._last_phase = phase
            self._last_change_step = int(step)
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
