"""Three foundational callbacks: EarlyStop, LRSchedule, Checkpoint.

These are the directly-shippable, low-risk callbacks. They don't
modify the model — they just decide *when* to act based on the
monitor's verdict.
"""

from __future__ import annotations

import pathlib
from typing import Any

from grokking_monitor import GrokkingMonitor

from .core import Action, BaseCallback, Phase


# ---------------------------------------------------------------------------
# 1) Early stop
# ---------------------------------------------------------------------------


class KairosEarlyStop(BaseCallback):
    """Stop training when the run looks like a memorisation-only dead end.

    Triggers ``stop_training=True`` when:
      * `train_acc >= memorisation_threshold` was reached at some step S, AND
      * `current_step - S >= stable_steps_to_abort`, AND
      * the monitor has NOT detected a grokking event by then.

    In other words: if we've been at train_acc = 1 with no test
    improvement for `stable_steps_to_abort` steps and Cassandra isn't
    signalling an imminent transition, abandon the run.

    Parameters
    ----------
    stable_steps_to_abort : int
        Steps after memorisation completes before giving up. Default
        10000 -- matches the slow-grok regime (real groks happen
        well within this).
    memorisation_threshold : float
        train_acc value at which "memorisation" is considered complete.
    min_step : int
        Don't fire earlier than this absolute step (sanity guard).
    """

    name = "KairosEarlyStop"

    def __init__(self, stable_steps_to_abort: int = 10_000,
                 memorisation_threshold: float = 0.99,
                 min_step: int = 1000) -> None:
        self.stable_steps_to_abort = int(stable_steps_to_abort)
        self.memorisation_threshold = float(memorisation_threshold)
        self.min_step = int(min_step)
        self._memorisation_step: int | None = None

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        train_acc = metrics.get("train_acc")
        if train_acc is None:
            return Action()
        if (self._memorisation_step is None
                and float(train_acc) >= self.memorisation_threshold):
            self._memorisation_step = int(step)
        if self._memorisation_step is None:
            return Action()
        if monitor.detected_event is not None:
            # Grokking already happened — never abort
            return Action()
        if step < self.min_step:
            return Action()
        if (step - self._memorisation_step) < self.stable_steps_to_abort:
            return Action()
        return Action(
            stop_training=True,
            stop_reason=(
                f"memorisation-only: train_acc >= "
                f"{self.memorisation_threshold} since step "
                f"{self._memorisation_step}; "
                f"{step - self._memorisation_step} steps without grokking"
            ),
        )


# ---------------------------------------------------------------------------
# 2) LR schedule
# ---------------------------------------------------------------------------


class KairosLRSchedule(BaseCallback):
    """Drop LR by ``drop_factor`` once at the grokking transition.

    Mirrors a common manual heuristic in grokking papers: when the
    test_loss CSD signature appears, the model is finding the
    generalising solution; too-high LR knocks it back. Reduce LR by
    ~10x at the transition to let it settle.

    Triggers a one-shot LR drop when the monitor's current phase is
    ``NEAR_CRITICAL`` or ``DRIFTING`` and we haven't yet dropped.

    Parameters
    ----------
    drop_factor : float
        Multiplicative LR scale (e.g. 0.1 = 10x reduction).
    optimizer : torch.optim.Optimizer | None
        If provided, the callback DIRECTLY scales each param group's
        ``lr``. Otherwise it just emits ``lr_multiplier`` in the Action
        and the training loop must apply it.
    """

    name = "KairosLRSchedule"

    def __init__(self, drop_factor: float = 0.1,
                 optimizer: Any | None = None) -> None:
        if not (0 < drop_factor < 1):
            raise ValueError(f"drop_factor must be in (0, 1); got {drop_factor}")
        self.drop_factor = float(drop_factor)
        self.optimizer = optimizer
        self._dropped: bool = False

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        if self._dropped:
            return Action()
        # Gate on the *confirmed* grokking event, not raw Cassandra
        # regime. On slow-grok runs the regime flips into `drifting`
        # thousands of steps before actual generalisation; dropping
        # LR that early froze training (validated 2026-05-15 head-to-
        # head: KAIROS-C with raw-regime gating got 0.154 final
        # test_acc vs BASELINE 0.365).
        if monitor.detected_event is None:
            return Action()
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = float(g["lr"]) * self.drop_factor
        self._dropped = True
        return Action(
            lr_multiplier=self.drop_factor,
            notes=[
                f"LR dropped by {self.drop_factor:.3f} at step {step} "
                f"(GrokkingMonitor confirmed grokking event)",
            ],
        )


# ---------------------------------------------------------------------------
# 3) Checkpoint at transition
# ---------------------------------------------------------------------------


class KairosCheckpoint(BaseCallback):
    """Save a model checkpoint when the monitor fires.

    Parameters
    ----------
    save_dir : str | pathlib.Path | None
        Directory to write ``torch.save`` checkpoints into. If None
        (and no ``save_fn``), the callback only flags
        ``save_checkpoint=True`` in the Action and the training loop
        is expected to handle it.
    save_fn : callable, optional
        Custom save function. Signature: ``save_fn(model, path)``.
        Useful for non-torch backends.
    filename_pattern : str
        Format string with ``{step}`` and ``{tag}`` placeholders.
        Default: ``"kairos_grok_step{step}_{tag}.pt"``.
    """

    name = "KairosCheckpoint"

    def __init__(self, save_dir: str | pathlib.Path | None = None,
                 save_fn=None,
                 filename_pattern: str = "kairos_grok_step{step}_{tag}.pt"
                 ) -> None:
        self.save_dir = pathlib.Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_fn = save_fn
        self.filename_pattern = str(filename_pattern)
        self._saved_event_step: int | None = None
        self._last_saved_path: pathlib.Path | None = None

    @property
    def last_saved_path(self) -> pathlib.Path | None:
        return self._last_saved_path

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        ev = monitor.detected_event
        if ev is None or self._saved_event_step is not None:
            return Action()
        # Monitor just fired — save once.
        self._saved_event_step = int(step)
        tag = ev.confidence
        if self.save_dir is not None:
            fname = self.filename_pattern.format(step=step, tag=tag)
            path = self.save_dir / fname
            model = metrics.get("model")
            if model is not None:
                if self.save_fn is not None:
                    self.save_fn(model, path)
                else:
                    try:
                        import torch
                        torch.save(model.state_dict(), path)
                    except Exception as e:
                        return Action(
                            save_checkpoint=True, checkpoint_tag=tag,
                            notes=[f"checkpoint save failed: {e}"],
                        )
                self._last_saved_path = path
                return Action(
                    save_checkpoint=True, checkpoint_tag=tag,
                    notes=[f"saved checkpoint to {path}"],
                )
        return Action(save_checkpoint=True, checkpoint_tag=tag,
                       notes=[f"grokking detected; please checkpoint (no save_dir)"])
