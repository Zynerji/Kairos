"""Framework integrations.

Each adapter forwards (step, metrics, model) into a CallbackBundle's
observe() and applies the returned Action against the framework's
control flow.
"""

from __future__ import annotations

from .lightning import KairosLightningCallback
from .hf import KairosHFCallback

__all__ = ["KairosLightningCallback", "KairosHFCallback"]
