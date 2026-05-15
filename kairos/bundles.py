"""Pre-wired CallbackBundles for common training regimes.

These are one-line factories so the typical user never has to know
about Cassandra, GrokkingMonitor, or which callback to instantiate.

Example
-------
>>> from kairos import recommended_bundle
>>> bundle = recommended_bundle("grokking", optimizer=opt,
...                              checkpoint_dir="./ckpt")
>>> for step in range(n_steps):
...     loss = train_step(...)
...     test_loss = eval(...)
...     bundle.observe(step, train_loss=loss, test_loss=test_loss,
...                     train_acc=ta, test_acc=tea)

Profiles
--------
- ``"grokking"`` — EarlyStop (long patience) + PendulumLR.for_grokking()
  + Checkpoint (saves on confirmed grokking + new test-acc highs)
- ``"distillation"`` — PendulumLR.for_distillation() + Checkpoint
  (train_loss driven; no early-stop since distillation is monotone)
- ``"pareto_post_training"`` — ParetoGuard with default Aletheia anchor
  + Checkpoint (multi-axis post-training)
- ``"growth_search"`` — GrowthController + LRSchedule + Checkpoint
  (architecture-growth experiments à la qGPT auto-grow)

All profiles return a ``CallbackBundle`` that consumes the standard
metric dict: ``train_loss / train_acc / test_loss / test_acc`` plus
profile-specific extras documented per-profile.
"""

from __future__ import annotations

from typing import Any

from .callbacks import KairosCheckpoint, KairosEarlyStop, KairosLRSchedule
from .core import CallbackBundle
from .growth_controller import KairosGrowthController
from .pareto_guard import KairosParetoGuard
from .pendulum_lr import KairosPendulumLR


def recommended_bundle(profile: str, *, optimizer: Any | None = None,
                        checkpoint_dir: str | None = None,
                        max_steps: int | None = None,
                        anchor: dict | None = None,
                        save_every: int = 0) -> CallbackBundle:
    """Build the recommended callback bundle for a training profile.

    Parameters
    ----------
    profile : {"grokking", "distillation", "pareto_post_training", "growth_search"}
        Which preset stack to assemble.
    optimizer : torch.optim.Optimizer | None
        Required for "grokking", "distillation", and "growth_search"
        (the PendulumLR/LRSchedule mutates ``param_groups[*]["lr"]``).
        Optional for "pareto_post_training" (no LR steering by default).
    checkpoint_dir : str | None
        Where ``KairosCheckpoint`` writes snapshots. If ``None``, no
        checkpoint callback is added.
    max_steps : int | None
        Hard step cap used by ``KairosEarlyStop`` for "grokking"
        profile. If ``None``, EarlyStop runs without a step cap.
    anchor : dict | None
        Required for "pareto_post_training". Maps axis name → minimum
        anchor value (e.g. ``{"acc": 0.5, "cal": 0.7}``).
    save_every : int
        If > 0, ``KairosCheckpoint`` also writes a snapshot every N
        steps regardless of phase. Default 0 (event-driven only).
    """
    profile = profile.lower().strip()
    if profile == "grokking":
        return _bundle_grokking(optimizer=optimizer,
                                 checkpoint_dir=checkpoint_dir,
                                 max_steps=max_steps,
                                 save_every=save_every)
    if profile == "distillation":
        return _bundle_distillation(optimizer=optimizer,
                                     checkpoint_dir=checkpoint_dir,
                                     save_every=save_every)
    if profile == "pareto_post_training":
        if anchor is None:
            raise ValueError(
                "pareto_post_training requires anchor={axis: float, ...}"
            )
        return _bundle_pareto(anchor=anchor,
                               checkpoint_dir=checkpoint_dir,
                               save_every=save_every)
    if profile == "growth_search":
        return _bundle_growth(optimizer=optimizer,
                                checkpoint_dir=checkpoint_dir,
                                save_every=save_every)
    if profile == "pretraining":
        return _bundle_pretraining(optimizer=optimizer,
                                     checkpoint_dir=checkpoint_dir,
                                     save_every=save_every)
    raise ValueError(
        f"unknown profile {profile!r}; expected one of "
        f"'grokking', 'distillation', 'pareto_post_training', "
        f"'growth_search', 'pretraining'"
    )


def _bundle_grokking(*, optimizer: Any | None,
                      checkpoint_dir: str | None,
                      max_steps: int | None,
                      save_every: int) -> CallbackBundle:
    del save_every  # KairosCheckpoint is event-driven
    callbacks: list = []
    callbacks.append(KairosPendulumLR.for_grokking(optimizer=optimizer))
    # EarlyStop tuned for slow groks: don't bail unless memorisation has
    # been parked for ~half the run with no Cassandra signal.
    stable = (max_steps // 2) if max_steps else 10_000
    callbacks.append(KairosEarlyStop(stable_steps_to_abort=stable))
    if checkpoint_dir is not None:
        callbacks.append(KairosCheckpoint(save_dir=checkpoint_dir))
    return CallbackBundle(*callbacks)


def _bundle_distillation(*, optimizer: Any | None,
                          checkpoint_dir: str | None,
                          save_every: int) -> CallbackBundle:
    del save_every
    callbacks: list = []
    callbacks.append(KairosPendulumLR.for_distillation(optimizer=optimizer))
    if checkpoint_dir is not None:
        callbacks.append(KairosCheckpoint(save_dir=checkpoint_dir))
    return CallbackBundle(*callbacks)


def _bundle_pareto(*, anchor: dict,
                    checkpoint_dir: str | None,
                    save_every: int) -> CallbackBundle:
    del save_every
    callbacks: list = [KairosParetoGuard(anchor=anchor, metric_prefix="")]
    if checkpoint_dir is not None:
        callbacks.append(KairosCheckpoint(save_dir=checkpoint_dir))
    return CallbackBundle(*callbacks)


def _bundle_growth(*, optimizer: Any | None,
                    checkpoint_dir: str | None,
                    save_every: int) -> CallbackBundle:
    del save_every
    callbacks: list = [
        KairosGrowthController(),
        KairosLRSchedule(optimizer=optimizer) if optimizer is not None
        else KairosLRSchedule(),
    ]
    if checkpoint_dir is not None:
        callbacks.append(KairosCheckpoint(save_dir=checkpoint_dir))
    return CallbackBundle(*callbacks)


def _bundle_pretraining(*, optimizer: Any | None,
                          checkpoint_dir: str | None,
                          save_every: int) -> CallbackBundle:
    """Pretraining profile: PendulumLR on train_loss (monotone descent),
    no early-stop (pretraining is checkpoint-driven), event-driven
    checkpoint snapshots if grokking/phase-shift IS detected (rare in
    LM pretraining, but Cassandra still fires on emergent-capability
    transitions per Power et al. 2022 follow-ups)."""
    del save_every
    callbacks: list = [
        KairosPendulumLR.for_distillation(optimizer=optimizer),
    ]
    if checkpoint_dir is not None:
        callbacks.append(KairosCheckpoint(save_dir=checkpoint_dir))
    return CallbackBundle(*callbacks)
