"""kairos: grokking-aware training-optimizer toolkit.

Catches the moment a neural network transitions from memorisation to
generalisation (via Grokking-Monitor + Cassandra) and acts on it:

   1. KairosEarlyStop         — abort runs that won't grok
   2. KairosLRSchedule        — one-shot LR drop at the confirmed event
   3. KairosCheckpoint        — snapshot model at the transition
   4. KairosSweepGate         — reallocate sweep compute to promising trials
   5. KairosAccelerator       — weight-noise pulses during the plateau (research)
   6. KairosCurriculum        — phase-aware optimizer settings (research)
   7. KairosProbe             — generic capability-emergence probe (research)
   8. KairosPendulumLR        — Hamiltonian-pendulum loss-driven LR adaptation
                                  (Kanon port; proven in Alembic DHART v14.2)
   9. KairosParetoGuard       — multi-axis Pareto-frontier rollback gate
                                  (Aletheia port; proven on Qwen3 9-axis post-training)
  10. KairosGrowthController  — saturation-triggered architecture-growth signal
                                  (qGPT-Infinity port; proven K=8→64 auto-grow)
"""

from __future__ import annotations

from . import _gm_path  # noqa: F401 -- side-effect: enables grokking_monitor import

from .core import (
    Action,
    BaseCallback,
    CallbackBundle,
    Phase,
    PhaseTransition,
)
from .callbacks import (
    KairosCheckpoint,
    KairosEarlyStop,
    KairosLRSchedule,
)
from .sweep import (
    KairosSweepGate,
    TrialSummary,
    TrialDecision,
)
from .acceleration import (
    KairosAccelerator,
)
from .curriculum import (
    KairosCurriculum,
    PhaseSettings,
)
from .probe import (
    KairosProbe,
    EmergenceReport,
)
from .pendulum_lr import KairosPendulumLR
from .pareto_guard import KairosParetoGuard, ParetoState
from .growth_controller import KairosGrowthController, GrowthSignal

__version__ = "0.2.0"

__all__ = [
    "Action",
    "BaseCallback",
    "CallbackBundle",
    "EmergenceReport",
    "GrowthSignal",
    "KairosAccelerator",
    "KairosCheckpoint",
    "KairosCurriculum",
    "KairosEarlyStop",
    "KairosGrowthController",
    "KairosLRSchedule",
    "KairosParetoGuard",
    "KairosPendulumLR",
    "KairosProbe",
    "KairosSweepGate",
    "ParetoState",
    "Phase",
    "PhaseSettings",
    "PhaseTransition",
    "TrialDecision",
    "TrialSummary",
]
