"""KairosGrowthController — saturation-triggered architecture-growth signal.

Ported from qGPT-Infinity's `training/growth_pendulum.py` (proven:
K=8 → 64 auto-grown, 6 successful events) + Overtone's `Growth
Pendulum` design notes.

Mechanism: three coupled pendulums (`theta_K`, `theta_W`, `theta_D`)
advance with angular velocities proportional to saturation signals:

  * `theta_K` — wave-mode / capacity saturation (signal: KL or
    loss-improvement stagnation over a rolling window)
  * `theta_W` — width / hidden-dim saturation (signal: hidden-state
    variance collapse)
  * `theta_D` — depth / layer-stack saturation (signal: cross-layer
    improvement diminishing)

Phase relationships determine growth direction via the bronze-mean
ratchet equation:

    R(t) = Σ_i κ_i · sin(β₃ · (θ_i − θ_train))

Whichever pendulum has the highest ratchet pressure when a saturation
threshold is breached recommends its dimension for growth.

This callback ONLY EMITS THE SIGNAL. It does NOT mutate the model.
The training loop reads ``Action.notes`` looking for entries of the
form ``"grow:<dim>"`` (where ``<dim>`` is one of ``K`` / ``W`` / ``D``)
and applies the surgery (`grow_k.py`, `grow_width.py`, `grow_depth.py`
in qGPT-Infinity terminology).

Design rationale for signal-only: weight-growth surgery is
model-architecture-specific and the right place for it is the
training loop. The Kairos callback owns *when* and *which*; the loop
owns *how*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from grokking_monitor import GrokkingMonitor

from .core import Action, BaseCallback


_BRONZE = (3.0 + math.sqrt(13.0)) / 2.0  # bronze metallic mean (β₃)


@dataclass
class GrowthSignal:
    """Snapshot of the growth controller's view of training.

    Attributes
    ----------
    step : int
    dim : str
        Which dimension was recommended (``"K"`` / ``"W"`` / ``"D"``),
        or ``"none"`` if no recommendation this step.
    ratchet_K, ratchet_W, ratchet_D : float
        Per-pendulum ratchet pressures.
    saturation_K, saturation_W, saturation_D : float
        Signal magnitudes (per-pendulum derivative of advancement).
    """
    step: int
    dim: str
    ratchet_K: float
    ratchet_W: float
    ratchet_D: float
    saturation_K: float
    saturation_W: float
    saturation_D: float


class KairosGrowthController(BaseCallback):
    """Three-pendulum architecture-growth signal generator.

    Parameters
    ----------
    K_threshold, W_threshold, D_threshold : float
        Minimum per-window improvement to NOT trigger growth in that
        dimension. Smaller = more permissive growth. Defaults from
        the qGPT-Infinity production config.
    kappa_K, kappa_W, kappa_D : float
        Pendulum coupling strengths. Default K > W > D (capacity grows
        fastest, depth slowest).
    window_size : int
        Rolling-window size for saturation signals.
    min_steps_before_grow : int
        Don't fire any growth signal before this many steps.
    cooldown_steps : int
        After firing a growth signal, wait at least this many steps
        before firing again.
    max_grow_events : int
        Hard cap on total growth signals.
    """

    name = "KairosGrowthController"

    def __init__(
        self,
        K_threshold: float = 0.05,
        W_threshold: float = 0.01,
        D_threshold: float = 0.02,
        kappa_K: float = 0.3,
        kappa_W: float = 0.2,
        kappa_D: float = 0.15,
        window_size: int = 100,
        min_steps_before_grow: int = 300,
        cooldown_steps: int = 500,
        max_grow_events: int = 16,
    ) -> None:
        self.K_threshold = float(K_threshold)
        self.W_threshold = float(W_threshold)
        self.D_threshold = float(D_threshold)
        self.kappa_K = float(kappa_K)
        self.kappa_W = float(kappa_W)
        self.kappa_D = float(kappa_D)
        self.window_size = int(window_size)
        self.min_steps_before_grow = int(min_steps_before_grow)
        self.cooldown_steps = int(cooldown_steps)
        self.max_grow_events = int(max_grow_events)
        # Pendulum angular states
        self.theta_K: float = 0.0
        self.theta_W: float = 0.0
        self.theta_D: float = 0.0
        self.theta_train: float = 0.0  # phase reference
        # Signal windows
        self._loss_hist: list[float] = []
        self._hidden_var_hist: list[float] = []
        self._depth_signal_hist: list[float] = []
        # Bookkeeping
        self._n_grow_events: int = 0
        self._last_grow_step: int | None = None
        self._history: list[GrowthSignal] = []

    @property
    def n_grow_events(self) -> int:
        return self._n_grow_events

    @property
    def history(self) -> list[GrowthSignal]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Saturation signals
    # ------------------------------------------------------------------

    def _signal_K(self) -> float:
        """Loss-improvement stagnation. Smaller = more saturated."""
        if len(self._loss_hist) < self.window_size:
            return 1.0  # no signal yet
        win = self._loss_hist[-self.window_size:]
        return float(win[0] - win[-1])  # positive if loss improving

    def _signal_W(self) -> float:
        """Hidden-variance collapse: |delta_var| over recent window."""
        if len(self._hidden_var_hist) < self.window_size:
            return 1.0
        win = self._hidden_var_hist[-self.window_size:]
        return float(abs(win[0] - win[-1]))

    def _signal_D(self) -> float:
        """Cross-layer improvement diminishing."""
        if len(self._depth_signal_hist) < self.window_size:
            return 1.0
        win = self._depth_signal_hist[-self.window_size:]
        return float(sum(win) / len(win))

    # ------------------------------------------------------------------
    # Pendulum dynamics
    # ------------------------------------------------------------------

    def _advance_pendulums(self, sig_K: float, sig_W: float, sig_D: float
                             ) -> None:
        """Each pendulum advances faster when its signal indicates
        saturation (small improvement)."""
        # Map "small signal -> large advancement"; clamp away from 0/inf
        def vel(sig: float, thr: float) -> float:
            return max(0.0, min(1.0, thr / max(abs(sig), 1e-6))) * 0.05
        self.theta_K += vel(sig_K, self.K_threshold)
        self.theta_W += vel(sig_W, self.W_threshold)
        self.theta_D += vel(sig_D, self.D_threshold)
        # theta_train advances at a steady reference rate
        self.theta_train += 0.02

    def _ratchets(self) -> tuple[float, float, float]:
        """Bronze-ratchet pressures per pendulum, relative to theta_train."""
        rK = self.kappa_K * math.sin(_BRONZE * (self.theta_K - self.theta_train))
        rW = self.kappa_W * math.sin(_BRONZE * (self.theta_W - self.theta_train))
        rD = self.kappa_D * math.sin(_BRONZE * (self.theta_D - self.theta_train))
        return rK, rW, rD

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def observe(self, step: int, monitor: GrokkingMonitor,
                **metrics: Any) -> Action:
        loss = metrics.get("train_loss")
        hidden_var = metrics.get("hidden_var")  # optional
        depth_signal = metrics.get("depth_signal")  # optional
        if loss is None:
            return Action()
        self._loss_hist.append(float(loss))
        # Default 1.0 for missing hidden/depth signals
        # (treated as "no saturation evidence" -> low growth pressure)
        self._hidden_var_hist.append(
            float(hidden_var) if hidden_var is not None else 1.0,
        )
        self._depth_signal_hist.append(
            float(depth_signal) if depth_signal is not None else 1.0,
        )

        sig_K, sig_W, sig_D = self._signal_K(), self._signal_W(), self._signal_D()
        self._advance_pendulums(sig_K, sig_W, sig_D)
        rK, rW, rD = self._ratchets()
        snap_dim = "none"
        notes: list[str] = []

        # Eligibility gates
        if (step >= self.min_steps_before_grow
                and self._n_grow_events < self.max_grow_events
                and (self._last_grow_step is None
                     or step - self._last_grow_step >= self.cooldown_steps)):
            # Trigger if ANY pendulum has positive ratchet pressure AND
            # its saturation signal is below threshold (truly saturated).
            candidates: list[tuple[float, str, float]] = []
            if rK > 0 and sig_K < self.K_threshold:
                candidates.append((rK, "K", sig_K))
            if rW > 0 and sig_W < self.W_threshold:
                candidates.append((rW, "W", sig_W))
            if rD > 0 and sig_D < self.D_threshold:
                candidates.append((rD, "D", sig_D))
            if candidates:
                # Pick the strongest ratchet
                candidates.sort(reverse=True)
                _, dim, sig = candidates[0]
                snap_dim = dim
                self._n_grow_events += 1
                self._last_grow_step = step
                notes.append(
                    f"grow:{dim}  ratchet={candidates[0][0]:.3f}  "
                    f"saturation={sig:.4f}  events={self._n_grow_events}"
                )

        self._history.append(GrowthSignal(
            step=int(step), dim=snap_dim,
            ratchet_K=rK, ratchet_W=rW, ratchet_D=rD,
            saturation_K=sig_K, saturation_W=sig_W, saturation_D=sig_D,
        ))

        return Action(notes=notes)
