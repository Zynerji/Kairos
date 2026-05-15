"""kairos: grokking-aware training-optimizer toolkit.

Catches the moment a neural network transitions from memorisation to
generalisation (via Grokking-Monitor + Cassandra) and acts on it:

  1. KairosEarlyStop      — abort runs that won't grok
  2. KairosLRSchedule     — drop LR at the transition
  3. KairosCheckpoint     — snapshot model at the transition
  4. KairosSweepGate      — reallocate sweep compute to promising trials
  5. KairosAccelerator    — weight-noise pulses during the plateau
  6. KairosCurriculum     — phase-aware optimizer settings
  7. KairosProbe          — generic capability-emergence probe
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

__version__ = "0.1.0"

__all__ = [
    "Action",
    "BaseCallback",
    "CallbackBundle",
    "EmergenceReport",
    "KairosAccelerator",
    "KairosCheckpoint",
    "KairosCurriculum",
    "KairosEarlyStop",
    "KairosLRSchedule",
    "KairosProbe",
    "KairosSweepGate",
    "Phase",
    "PhaseSettings",
    "PhaseTransition",
    "TrialDecision",
    "TrialSummary",
]
