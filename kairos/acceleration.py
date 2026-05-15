"""KairosAccelerator: weight-noise pulses during the plateau.

Research-grade. Published work (Liu et al. 2022; Nanda 2023) suggests
that small Gaussian perturbations to the weights during the
post-memorisation plateau can accelerate the grokking transition,
often by an order of magnitude.

Triggers a perturbation when:
  * `train_acc >= memorisation_threshold` for at least `wait_steps`, AND
  * The monitor has not yet detected a grokking event, AND
  * It has been at least `cooldown_steps` since the last pulse.

The Action's `inject_noise_sigma` carries the recommended std-dev.
If the optimizer is provided AND `auto_apply=True`, the callback
directly perturbs every parameter in-place (one-time).
"""

from __future__ import annotations

from typing import Any

from grokking_monitor import GrokkingMonitor

from .core import Action, BaseCallback


class KairosAccelerator(BaseCallback):
    """Inject weight-noise pulses to accelerate grokking.

    Parameters
    ----------
    sigma : float
        Noise std-dev as a fraction of each parameter's RMS scale.
        Default 0.01 (1%). Too high: destroys learned features.
        Too low: no effect.
    wait_steps : int
        How many post-memorisation steps to wait before the first
        pulse.
    cooldown_steps : int
        Minimum steps between successive pulses.
    max_pulses : int
        Hard cap on number of pulses.
    memorisation_threshold : float
        train_acc threshold for "memorisation complete".
    auto_apply : bool
        If True and a `model` is passed to `observe()`, perturb its
        parameters in-place. Otherwise the Action's
        `inject_noise_sigma` is the only signal.
    """

    name = "KairosAccelerator"

    def __init__(self, sigma: float = 0.01, wait_steps: int = 500,
                 cooldown_steps: int = 500, max_pulses: int = 20,
                 memorisation_threshold: float = 0.99,
                 auto_apply: bool = True) -> None:
        if not (0 < sigma < 1):
            raise ValueError(f"sigma must be in (0, 1); got {sigma}")
        self.sigma = float(sigma)
        self.wait_steps = int(wait_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.max_pulses = int(max_pulses)
        self.memorisation_threshold = float(memorisation_threshold)
        self.auto_apply = bool(auto_apply)
        self._memorisation_step: int | None = None
        self._last_pulse_step: int | None = None
        self._n_pulses: int = 0

    @property
    def n_pulses(self) -> int:
        return self._n_pulses

    def _perturb(self, model: Any) -> None:
        """Apply Gaussian noise scaled to each parameter's RMS."""
        try:
            import torch
        except ImportError:
            return
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                rms = float(p.detach().pow(2).mean().sqrt().item())
                if rms == 0:
                    continue
                noise = torch.randn_like(p) * (rms * self.sigma)
                p.add_(noise)

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        if monitor.detected_event is not None:
            return Action()  # transition already happened; don't perturb
        if self._n_pulses >= self.max_pulses:
            return Action()
        train_acc = metrics.get("train_acc")
        if train_acc is None:
            return Action()
        if self._memorisation_step is None:
            if float(train_acc) >= self.memorisation_threshold:
                self._memorisation_step = int(step)
            return Action()
        # Have memorised; check timing
        if step - self._memorisation_step < self.wait_steps:
            return Action()
        if (self._last_pulse_step is not None
                and step - self._last_pulse_step < self.cooldown_steps):
            return Action()

        # Fire pulse
        self._last_pulse_step = int(step)
        self._n_pulses += 1
        model = metrics.get("model")
        applied = False
        if self.auto_apply and model is not None:
            self._perturb(model)
            applied = True
        return Action(
            inject_noise_sigma=self.sigma,
            notes=[
                f"accelerator pulse #{self._n_pulses} at step {step}"
                + (" (applied)" if applied else ""),
            ],
        )
