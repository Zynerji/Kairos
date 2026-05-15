"""Core data structures + base callback class for Kairos.

Every Kairos component returns an ``Action`` on each call to
``observe(...)``. ``CallbackBundle`` aggregates multiple components
and merges their actions.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from grokking_monitor import GrokkingEvent, GrokkingMonitor


class Phase(str, enum.Enum):
    """Training phase classification."""
    MEMORISING = "memorising"          # train_acc rising, test_acc near chance
    PLATEAU = "plateau"                # train_acc ~ 1, test_acc still near chance
    NEAR_CRITICAL = "near_critical"    # CSD signature; transition imminent
    DRIFTING = "drifting"              # test metrics moving toward generalisation
    POST = "post"                      # transition complete; test_acc high
    UNKNOWN = "unknown"


@dataclass
class PhaseTransition:
    """A recorded change in `Phase` at a specific step."""
    step: int
    from_phase: Phase
    to_phase: Phase


@dataclass
class Action:
    """Aggregated decision from one or more Kairos components.

    A single component never sets `stop_training=True` unless it has
    decided the run is dead. The bundle's ``observe`` returns a
    merged ``Action`` per step that the training loop should respect.
    """
    stop_training: bool = False
    stop_reason: str = ""
    lr_multiplier: float = 1.0          # multiply current LR by this if != 1.0
    weight_decay_multiplier: float = 1.0
    save_checkpoint: bool = False
    checkpoint_tag: str = ""
    inject_noise_sigma: float = 0.0     # >0 => caller should perturb weights this much
    phase: Phase = Phase.UNKNOWN
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "Action") -> "Action":
        """Combine with another Action (commutative-ish)."""
        return Action(
            stop_training=self.stop_training or other.stop_training,
            stop_reason=(self.stop_reason or other.stop_reason),
            # Multiplicative effects compose
            lr_multiplier=self.lr_multiplier * other.lr_multiplier,
            weight_decay_multiplier=(
                self.weight_decay_multiplier * other.weight_decay_multiplier
            ),
            save_checkpoint=self.save_checkpoint or other.save_checkpoint,
            checkpoint_tag=(self.checkpoint_tag or other.checkpoint_tag),
            inject_noise_sigma=max(self.inject_noise_sigma, other.inject_noise_sigma),
            phase=(other.phase if other.phase != Phase.UNKNOWN else self.phase),
            notes=self.notes + other.notes,
        )


class BaseCallback:
    """All Kairos components inherit from this.

    Subclasses implement ``observe(step, monitor, **metrics) -> Action``.
    They typically read from the shared monitor, never mutate it.
    """

    name: str = "BaseCallback"

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: float) -> Action:
        return Action()


def _classify_phase(monitor: GrokkingMonitor, train_acc: float | None,
                    test_acc: float | None) -> Phase:
    """Map current GrokkingMonitor state + raw metrics to a Phase."""
    if monitor.detected_event is not None:
        return Phase.POST
    # If we have an explicit Cassandra diagnosis, prefer it
    state = monitor.state()
    diag = state.last_cassandra_diagnosis
    if diag is not None:
        regime = diag.get("regime")
        if regime == "near_critical":
            return Phase.NEAR_CRITICAL
        if regime == "drifting":
            return Phase.DRIFTING
        if regime == "post":
            return Phase.POST
    # Fall back to direct metric inspection
    if train_acc is None or test_acc is None:
        return Phase.UNKNOWN
    if train_acc < 0.5:
        return Phase.MEMORISING
    if train_acc >= 0.95 and test_acc < 0.5:
        return Phase.PLATEAU
    if test_acc >= 0.95:
        return Phase.POST
    return Phase.DRIFTING


class CallbackBundle:
    """Aggregates multiple Kairos components into a single observe() call.

    Owns the shared `GrokkingMonitor`. Each component receives a read-
    only reference to it. The bundle returns the merged ``Action``.

    Parameters
    ----------
    *callbacks : BaseCallback
        Any number of Kairos components.
    monitor : GrokkingMonitor, optional
        Pre-constructed monitor. If omitted, uses the slow-grok preset
        — which is the right default for real CPU/GPU training runs.
    """

    def __init__(self, *callbacks: BaseCallback,
                 monitor: GrokkingMonitor | None = None) -> None:
        self.callbacks = list(callbacks)
        self.monitor = monitor or GrokkingMonitor.for_slow_grok(check_every=200)
        self._phase_history: list[PhaseTransition] = []
        self._last_phase = Phase.UNKNOWN

    def observe(self, step: int, **metrics: Any) -> Action:
        """Feed the monitor + every callback. Returns the merged Action.

        The keyword arguments are forwarded to the monitor (`train_loss`,
        `train_acc`, `test_loss`, `test_acc`) and to every callback
        (so e.g. `model` can be passed through to checkpointers).
        """
        # 1) Update the shared monitor
        primitive_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k in {
                "train_loss", "train_acc", "test_loss", "test_acc",
            }
        }
        self.monitor.observe(step, **primitive_metrics)

        # 2) Track phase transitions
        phase = _classify_phase(
            self.monitor,
            primitive_metrics.get("train_acc"),
            primitive_metrics.get("test_acc"),
        )
        if phase != self._last_phase and phase != Phase.UNKNOWN:
            self._phase_history.append(
                PhaseTransition(step=step, from_phase=self._last_phase, to_phase=phase),
            )
            self._last_phase = phase

        # 3) Run every callback
        action = Action(phase=phase)
        for cb in self.callbacks:
            sub = cb.observe(step, self.monitor, **metrics)
            sub.phase = phase
            action = action.merge(sub)
        return action

    @property
    def phase_history(self) -> list[PhaseTransition]:
        return list(self._phase_history)

    @property
    def current_phase(self) -> Phase:
        return self._last_phase
