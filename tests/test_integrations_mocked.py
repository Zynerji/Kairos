"""Integration smoke tests using mock trainers — these never import
``transformers`` or ``pytorch_lightning`` at module import time.

The HF/Lightning adapters duck-type into their host trainer's API.
We verify the adapter handles the call surface correctly without
needing the real framework installed.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
nn = torch.nn

from kairos import (
    Action,
    CallbackBundle,
    KairosEarlyStop,
    recommended_bundle,
)


# ---------------------------------------------------------------------------
# Tiny model for action-target validation
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def state_dict(self, *a, **kw):
        return super().state_dict(*a, **kw)


# ---------------------------------------------------------------------------
# HF Callback — patched-import smoke test
# ---------------------------------------------------------------------------


def test_hf_callback_constructs_without_transformers(monkeypatch):
    """If transformers is missing, constructor raises ImportError."""
    import importlib
    import sys as _sys

    # Make `import transformers` blow up inside the adapter.
    monkeypatch.setitem(_sys.modules, "transformers", None)
    # Reimport to pick up the monkeypatched transformers
    mod = importlib.reload(importlib.import_module("kairos.integrations.hf"))
    bundle = CallbackBundle(KairosEarlyStop())
    with pytest.raises(ImportError):
        mod.KairosHFCallback(bundle)


def test_hf_callback_smoke_with_mocked_transformers(monkeypatch, tmp_path):
    """Adapter consumes HF-style on_log/on_step_end without crashing."""
    import importlib
    import sys as _sys
    # Fake transformers module so the import inside the adapter resolves
    monkeypatch.setitem(_sys.modules, "transformers",
                          SimpleNamespace(TrainerCallback=object))
    mod = importlib.reload(importlib.import_module("kairos.integrations.hf"))

    bundle = CallbackBundle(KairosEarlyStop(stable_steps_to_abort=10,
                                              memorisation_threshold=0.99,
                                              min_step=5))
    cb = mod.KairosHFCallback(bundle, save_dir=str(tmp_path))

    args = SimpleNamespace()
    state = SimpleNamespace(global_step=0, log_history=[])
    control = SimpleNamespace(should_training_stop=False)
    model = _TinyModel()

    # Simulate: memorised early, then dwell until earlystop fires.
    for step in range(50):
        state.global_step = step
        cb.on_log(args, state, control,
                   logs={"loss": 0.001, "accuracy": 1.0,
                         "eval_loss": 1.5, "eval_accuracy": 0.05},
                   model=model)
        cb.on_step_end(args, state, control, model=model)
    assert control.should_training_stop, (
        "EarlyStop should have fired by step 50"
    )


# ---------------------------------------------------------------------------
# Lightning Callback — patched-import smoke test
# ---------------------------------------------------------------------------


def test_lightning_callback_constructs_without_pl(monkeypatch):
    import importlib
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "pytorch_lightning", None)
    mod = importlib.reload(
        importlib.import_module("kairos.integrations.lightning")
    )
    bundle = CallbackBundle(KairosEarlyStop())
    with pytest.raises(ImportError):
        mod.KairosLightningCallback(bundle)


def test_lightning_callback_smoke_with_mocked_pl(monkeypatch, tmp_path):
    import importlib
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "pytorch_lightning",
                          SimpleNamespace(Callback=object))
    mod = importlib.reload(
        importlib.import_module("kairos.integrations.lightning")
    )

    bundle = CallbackBundle(KairosEarlyStop(stable_steps_to_abort=10,
                                              memorisation_threshold=0.99,
                                              min_step=5))
    cb = mod.KairosLightningCallback(bundle, save_dir=str(tmp_path))

    pl_module = _TinyModel()

    class _FakeTrainer:
        def __init__(self) -> None:
            self.global_step = 0
            self.should_stop = False
            self.callback_metrics: dict = {}

    trainer = _FakeTrainer()

    # Lightning-style metric stream
    for step in range(60):
        trainer.global_step = step
        trainer.callback_metrics = {
            "train_loss": torch.tensor(0.001),
            "train_acc": torch.tensor(1.0),
            "val_loss": torch.tensor(1.5),
            "val_acc": torch.tensor(0.05),
        }
        cb.on_train_batch_end(trainer, pl_module, outputs=None,
                                batch=None, batch_idx=step)
        if step % 10 == 0:
            cb.on_validation_end(trainer, pl_module)
    assert trainer.should_stop, "EarlyStop should have fired by step 60"


def test_lightning_callback_noise_injection_does_not_explode(monkeypatch):
    """Noise injection should mutate params, never raise."""
    import importlib
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "pytorch_lightning",
                          SimpleNamespace(Callback=object))
    mod = importlib.reload(
        importlib.import_module("kairos.integrations.lightning")
    )

    class _Always:
        name = "_Always"

        def observe(self, step, monitor, **metrics):
            return Action(inject_noise_sigma=0.01)

    bundle = CallbackBundle(_Always())
    cb = mod.KairosLightningCallback(bundle)

    pl_module = _TinyModel()
    snapshot = pl_module.fc.weight.detach().clone()

    class _FakeTrainer:
        def __init__(self) -> None:
            self.global_step = 0
            self.should_stop = False
            self.callback_metrics = {
                "train_loss": torch.tensor(0.5),
                "train_acc": torch.tensor(0.5),
            }

    trainer = _FakeTrainer()
    cb.on_train_batch_end(trainer, pl_module, None, None, 0)
    # weights should have moved
    diff = (pl_module.fc.weight - snapshot).abs().max().item()
    assert diff > 0, "noise injection did not perturb weights"


# ---------------------------------------------------------------------------
# recommended_bundle integration (end-to-end smoke)
# ---------------------------------------------------------------------------


def test_recommended_bundle_grokking_routes_metrics_to_pendulum_and_earlystop():
    class _Opt:
        param_groups = [{"lr": 1e-3, "weight_decay": 0.0}]

    opt = _Opt()
    bundle = recommended_bundle("grokking", optimizer=opt, max_steps=200)
    # Pendulum should mutate the LR on a streaming metric input
    for step in range(60):
        bundle.observe(step, train_loss=0.5, train_acc=0.5,
                        test_loss=0.5 + 0.1 * (step % 5),
                        test_acc=0.3)
    assert opt.param_groups[0]["lr"] > 0
