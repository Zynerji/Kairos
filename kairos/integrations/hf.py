"""HuggingFace Trainer Callback adapter.

Wraps a Kairos CallbackBundle as a `transformers.TrainerCallback`.
Reads `state.log_history` for the latest train/eval metrics; applies
the Action against `control` (early stop) and `model` (perturbation,
checkpoint).
"""

from __future__ import annotations

import pathlib
from typing import Any

from ..core import CallbackBundle


def _trainer_callback_base():
    """Return ``transformers.TrainerCallback`` if available, else ``object``.

    We subclass the real ``TrainerCallback`` when transformers is
    installed so HF's strict 5.x callback handler (which calls
    ``getattr(callback, event)`` for every event including
    ``on_init_end``) finds the no-op defaults it expects.
    """
    try:
        from transformers import TrainerCallback
        return TrainerCallback
    except Exception:
        return object


class KairosHFCallback(_trainer_callback_base()):
    """HuggingFace TrainerCallback adapter.

    Parameters
    ----------
    bundle : CallbackBundle
    train_loss_key, eval_loss_key, train_acc_key, eval_acc_key : str
        HF log-history keys.
    save_dir : str | pathlib.Path | None
    """

    def __init__(
        self,
        bundle: CallbackBundle,
        train_loss_key: str = "loss",
        eval_loss_key: str = "eval_loss",
        train_acc_key: str = "accuracy",
        eval_acc_key: str = "eval_accuracy",
        save_dir: str | pathlib.Path | None = None,
    ) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "KairosHFCallback needs transformers; "
                "install with `pip install transformers`",
            ) from e
        # If our base is TrainerCallback, call its __init__ to set up
        # any state it tracks.
        try:
            super().__init__()
        except TypeError:
            pass
        self.bundle = bundle
        self.train_loss_key = train_loss_key
        self.eval_loss_key = eval_loss_key
        self.train_acc_key = train_acc_key
        self.eval_acc_key = eval_acc_key
        self.save_dir = pathlib.Path(save_dir) if save_dir is not None else None
        self._last: dict[str, float] = {}

    # Duck-typed HF hooks. The real TrainerCallback has on_log,
    # on_evaluate, on_step_end. We override the ones we need.

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        logs = logs or {}
        if self.train_loss_key in logs:
            self._last["train_loss"] = float(logs[self.train_loss_key])
        if self.train_acc_key in logs:
            self._last["train_acc"] = float(logs[self.train_acc_key])
        if self.eval_loss_key in logs:
            self._last["test_loss"] = float(logs[self.eval_loss_key])
        if self.eval_acc_key in logs:
            self._last["test_acc"] = float(logs[self.eval_acc_key])

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if not self._last:
            return control
        action = self.bundle.observe(step, model=model, **self._last)
        if action.stop_training:
            control.should_training_stop = True
        if action.save_checkpoint and self.save_dir is not None and model is not None:
            try:
                import torch
                self.save_dir.mkdir(parents=True, exist_ok=True)
                p = self.save_dir / f"kairos_grok_step{step}_{action.checkpoint_tag}.pt"
                torch.save(model.state_dict(), p)
            except Exception:
                pass
        if action.inject_noise_sigma > 0 and model is not None:
            try:
                import torch
                sigma = action.inject_noise_sigma
                with torch.no_grad():
                    for p in model.parameters():
                        if not p.requires_grad:
                            continue
                        rms = float(p.detach().pow(2).mean().sqrt().item())
                        if rms == 0:
                            continue
                        p.add_(torch.randn_like(p) * (rms * sigma))
            except Exception:
                pass
        return control
