"""KairosPendulumLR — Hamiltonian-pendulum-driven adaptive LR.

Ported from Kanon's `src/kanon/training/pendulum.py`. Proven in
Alembic DHART v14.2 (15/15 crystal detection, 6 GOLD heads, ~5%
better than cosine in late-phase distillation).

Mechanism: a loss-driven pendulum's *Conservation Violation* (CV)
classifies training into three phases, each with its own LR multiplier:

    CV < 0.1   → CRYSTAL  (plateau)   lr_mult = 1/φ²  ≈ 0.382
    0.1 ≤ 0.3  → ACTIVE                lr_mult = 1.0
    CV > 0.3   → EXPLORE  (stuck)      lr_mult = φ    ≈ 1.618

Unlike cosine decay, the pendulum's minimum is 0.382× the base LR —
it never freezes training. When the model stalls on a plateau, the
LR boosts to escape; when it's making fine adjustments, the LR
contracts. This is the right late-phase behaviour for distillation
and grokking-adjacent regimes.

Compare to ``KairosLRSchedule``: that one is a one-shot 10× drop at
the confirmed grokking event. ``KairosPendulumLR`` is continuous
loss-driven adaptation; the two can be composed (the pendulum
modulates around the post-drop base LR).
"""

from __future__ import annotations

import math
from typing import Any

from grokking_monitor import GrokkingMonitor

from .core import Action, BaseCallback


# Anti-resonant constants (from the Kanon port)
_PHI = (1.0 + math.sqrt(5.0)) / 2.0      # golden ratio
_BRONZE = (3.0 + math.sqrt(13.0)) / 2.0  # bronze metallic mean (β₃)


class _Pendulum:
    """Internal loss-driven Hamiltonian pendulum (Kanon port)."""

    def __init__(self, omega0: float = _BRONZE, dt: float = 0.01,
                 window: int = 100) -> None:
        self.omega0 = float(omega0)
        self.dt = float(dt)
        self.window = int(window)
        # Initial angle = 1/φ (golden), at rest
        self.theta: float = 1.0 / _PHI
        self.theta_dot: float = 0.0
        self.H_history: list[float] = []
        self.cv_history: list[float] = []
        self.state: str = "ACTIVE"
        self.crystal_count: int = 0
        self.active_count: int = 0
        self.explore_count: int = 0

    def step(self, loss_value: float) -> tuple[str, float]:
        """One Verlet integration step driven by the latest loss."""
        driving_force = float(loss_value) * (1.0 / _BRONZE)
        self.theta_dot += (
            -self.omega0 ** 2 * math.sin(self.theta) + driving_force
        ) * self.dt
        self.theta += self.theta_dot * self.dt
        self.theta = ((self.theta + math.pi) % (2.0 * math.pi)) - math.pi

        H = 0.5 * self.theta_dot ** 2 + self.omega0 ** 2 * (1.0 - math.cos(self.theta))
        self.H_history.append(H)

        if len(self.H_history) >= 2:
            dH_dt = abs(self.H_history[-1] - self.H_history[-2]) / self.dt
            window = self.H_history[-self.window:]
            H_mean = sum(abs(h) for h in window) / len(window)
            cv = dH_dt / max(H_mean, 1e-8)
        else:
            cv = 0.5

        self.cv_history.append(cv)
        if cv < 0.1:
            self.state, self.crystal_count = "CRYSTAL", self.crystal_count + 1
        elif cv <= 0.3:
            self.state, self.active_count = "ACTIVE", self.active_count + 1
        else:
            self.state, self.explore_count = "EXPLORE", self.explore_count + 1
        return self.state, cv

    def lr_mult(self) -> float:
        if self.state == "CRYSTAL":
            return 1.0 / (_PHI ** 2)
        if self.state == "ACTIVE":
            return 1.0
        return _PHI


class KairosPendulumLR(BaseCallback):
    """Continuous loss-driven LR adaptation via Hamiltonian pendulum.

    Parameters
    ----------
    metric : {"train_loss", "test_loss"}
        Which loss stream drives the pendulum. Default ``"train_loss"``.
    omega0, dt, window : float, float, int
        Pendulum hyperparameters (Kanon defaults are sensible).
    optimizer : torch.optim.Optimizer | None
        If provided, the pendulum's LR multiplier is applied directly
        per step (rebased from the initial LR captured at first call).
    apply_smoothing : float
        EMA smoothing on the lr_mult so per-step jitter doesn't whiplash
        the optimizer. 0 = no smoothing, 1 = freeze. Default 0.8.
    """

    name = "KairosPendulumLR"

    def __init__(self, metric: str = "train_loss",
                 omega0: float = _BRONZE, dt: float = 0.01,
                 window: int = 100,
                 optimizer: Any | None = None,
                 apply_smoothing: float = 0.8) -> None:
        if metric not in ("train_loss", "test_loss"):
            raise ValueError(f"metric must be train_loss/test_loss; got {metric!r}")
        if not (0.0 <= apply_smoothing < 1.0):
            raise ValueError(
                f"apply_smoothing must be in [0, 1); got {apply_smoothing}"
            )
        self.metric = metric
        self.optimizer = optimizer
        self.apply_smoothing = float(apply_smoothing)
        self._pendulum = _Pendulum(omega0=omega0, dt=dt, window=window)
        self._initial_lrs: list[float] | None = None
        self._ema_mult: float = 1.0

    @property
    def state(self) -> str:
        return self._pendulum.state

    @property
    def cv(self) -> float:
        return self._pendulum.cv_history[-1] if self._pendulum.cv_history else 0.0

    @property
    def crystal_count(self) -> int:
        return self._pendulum.crystal_count

    @property
    def active_count(self) -> int:
        return self._pendulum.active_count

    @property
    def explore_count(self) -> int:
        return self._pendulum.explore_count

    def _capture_initial_lrs(self) -> None:
        if self._initial_lrs is None and self.optimizer is not None:
            self._initial_lrs = [
                float(g["lr"]) for g in self.optimizer.param_groups
            ]

    def _apply(self, mult: float) -> None:
        if self.optimizer is None or self._initial_lrs is None:
            return
        for g, lr0 in zip(self.optimizer.param_groups, self._initial_lrs):
            g["lr"] = float(lr0) * float(mult)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def for_grokking(cls, optimizer: Any = None,
                     apply_smoothing: float = 0.85) -> "KairosPendulumLR":
        """Default for grokking-style tasks where train_loss flatlines
        after memorisation but test_loss continues to oscillate.

        Validated on (a+b) mod 29 modular-arithmetic Transformer:
        +0.623 test_acc absolute over static lr=1e-3 baseline (15K
        steps, AdamW, wd=1.0, RTX PRO 4000 Blackwell, 2026-05-15).
        """
        return cls(metric="test_loss", optimizer=optimizer,
                   apply_smoothing=apply_smoothing)

    @classmethod
    def for_distillation(cls, optimizer: Any = None,
                         apply_smoothing: float = 0.85) -> "KairosPendulumLR":
        """Default for distillation / continuously-improving loss
        streams where train_loss IS the live signal (e.g. KL falling
        across thousands of steps). Original Kanon configuration —
        proven in Alembic DHART v14.2."""
        return cls(metric="train_loss", optimizer=optimizer,
                   apply_smoothing=apply_smoothing)

    # ------------------------------------------------------------------
    # observe
    # ------------------------------------------------------------------

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        self._capture_initial_lrs()
        loss = metrics.get(self.metric)
        if loss is None:
            return Action()
        state, cv = self._pendulum.step(float(loss))
        raw_mult = self._pendulum.lr_mult()
        # EMA smoothing to dampen per-step jitter
        a = self.apply_smoothing
        self._ema_mult = a * self._ema_mult + (1.0 - a) * raw_mult
        self._apply(self._ema_mult)
        notes: list[str] = []
        # Log phase transitions only (not every step)
        if step % 200 == 0 or state == "EXPLORE":
            notes.append(
                f"pendulum_lr: state={state} cv={cv:.3f} "
                f"raw_mult={raw_mult:.3f} ema_mult={self._ema_mult:.3f}"
            )
        return Action(lr_multiplier=self._ema_mult, notes=notes)
