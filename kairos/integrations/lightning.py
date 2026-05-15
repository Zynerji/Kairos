"""PyTorch Lightning Callback adapter.

Wraps a Kairos CallbackBundle as a `pytorch_lightning.Callback`. Hooks
into `on_train_batch_end` (per-step metrics) and `on_validation_end`
(test metrics). Applies the bundle's Action against the trainer:

  * stop_training -> trainer.should_stop = True
  * save_checkpoint -> trainer.save_checkpoint(...)
  * inject_noise_sigma -> apply to pl_module.parameters() in-place
  * lr/wd already mutated by the schedule callbacks via optimizer ref
"""

from __future__ import annotations

import pathlib
from typing import Any

from ..core import CallbackBundle


class KairosLightningCallback:
    """Lightning callback adapter for Kairos.

    Lightning is an optional dependency; this class only requires
    pytorch_lightning when actually instantiated against a trainer.

    Parameters
    ----------
    bundle : CallbackBundle
    train_metric_keys : dict
        Maps Lightning metric names to Kairos names. Default:
        ``{"train_loss": "train_loss", "train_acc": "train_acc",
           "val_loss": "test_loss", "val_acc": "test_acc"}``.
    save_dir : str | pathlib.Path | None
        Optional override for KairosCheckpoint save_dir.
    """

    def __init__(
        self,
        bundle: CallbackBundle,
        train_metric_keys: dict | None = None,
        save_dir: str | pathlib.Path | None = None,
    ) -> None:
        try:
            import pytorch_lightning as pl  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "KairosLightningCallback needs pytorch_lightning; "
                "install with `pip install pytorch-lightning`",
            ) from e
        self.bundle = bundle
        self.train_metric_keys = train_metric_keys or {
            "train_loss": "train_loss", "train_acc": "train_acc",
            "val_loss": "test_loss", "val_acc": "test_acc",
        }
        self.save_dir = pathlib.Path(save_dir) if save_dir is not None else None
        self._last_metrics: dict[str, float] = {}

    # Lightning hooks (duck-typed; we don't subclass to avoid hard
    # dependency on pytorch_lightning at import time)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try:
            cm = dict(trainer.callback_metrics)
        except Exception:
            cm = {}
        metrics = {}
        for src, dst in self.train_metric_keys.items():
            if src in cm:
                try:
                    metrics[dst] = float(cm[src].item() if hasattr(cm[src], "item") else cm[src])
                except Exception:
                    pass
        self._last_metrics.update(metrics)
        step = int(trainer.global_step)
        action = self.bundle.observe(step, model=pl_module, **self._last_metrics)
        self._apply_action(trainer, pl_module, action, step)

    def on_validation_end(self, trainer, pl_module):
        try:
            cm = dict(trainer.callback_metrics)
        except Exception:
            cm = {}
        for src, dst in self.train_metric_keys.items():
            if src in cm:
                try:
                    self._last_metrics[dst] = float(
                        cm[src].item() if hasattr(cm[src], "item") else cm[src]
                    )
                except Exception:
                    pass

    def _apply_action(self, trainer, pl_module, action, step):
        if action.stop_training:
            trainer.should_stop = True
        if action.save_checkpoint and self.save_dir is not None:
            try:
                import torch
                p = self.save_dir / f"kairos_grok_step{step}_{action.checkpoint_tag}.pt"
                self.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(pl_module.state_dict(), p)
            except Exception:
                pass
        if action.inject_noise_sigma > 0:
            try:
                import torch
                with torch.no_grad():
                    sigma = action.inject_noise_sigma
                    for p in pl_module.parameters():
                        if not p.requires_grad:
                            continue
                        rms = float(p.detach().pow(2).mean().sqrt().item())
                        if rms == 0:
                            continue
                        p.add_(torch.randn_like(p) * (rms * sigma))
            except Exception:
                pass
