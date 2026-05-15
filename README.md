# Kairos

> *καιρός* — Greek: "the right moment, the critical time"

**Grokking-aware training-optimizer toolkit.** Catches the moment a
neural network transitions from memorisation to generalisation and
acts on it.

Built on [Grokking-Monitor](https://github.com/Zynerji/Grokking-Monitor)
and [Cassandra](https://github.com/Zynerji/Cassandra).

## The seven components

| # | Component | What it does | When to use |
|---|---|---|---|
| 1 | `KairosEarlyStop`    | Abort runs that won't grok (memorisation-only) | Hyperparameter sweeps, expensive runs |
| 2 | `KairosLRSchedule`   | Drop LR ~10× at the grokking transition | Almost always |
| 3 | `KairosCheckpoint`   | Snapshot model at the transition | Always — you want this checkpoint |
| 4 | `KairosSweepGate`    | Kill losing trials at step K | Multi-trial hparam search |
| 5 | `KairosAccelerator`  | Weight-noise pulses during the plateau (research) | When grokking is too slow |
| 6 | `KairosCurriculum`   | Phase-aware lr / wd settings | When you want a phase-driven schedule |
| 7 | `KairosProbe`        | Generic capability-emergence probe | Tracking many emergent skills in parallel |

## Quick start: minimal training loop

```python
from kairos import (
    CallbackBundle, KairosEarlyStop, KairosLRSchedule,
    KairosCheckpoint, KairosCurriculum, KairosAccelerator,
)

bundle = CallbackBundle(
    KairosEarlyStop(stable_steps_to_abort=20_000, min_step=1000),
    KairosLRSchedule(drop_factor=0.1, optimizer=opt),
    KairosCheckpoint(save_dir="checkpoints/"),
    KairosAccelerator(sigma=0.005, wait_steps=2000, max_pulses=8),
    KairosCurriculum(optimizer=opt),
)

for step in range(n_steps):
    train_loss, train_acc, test_loss, test_acc = train_step(...)
    action = bundle.observe(
        step,
        train_loss=train_loss, train_acc=train_acc,
        test_loss=test_loss, test_acc=test_acc,
        model=model,
    )
    if action.stop_training:
        print(f"Aborted: {action.stop_reason}")
        break
```

The `Action` returned by `observe()` carries every decision the
components reached — `stop_training`, `lr_multiplier`,
`weight_decay_multiplier`, `save_checkpoint`, `inject_noise_sigma`,
plus `phase` and human-readable `notes`.

## Framework integrations

### HuggingFace Transformers

```python
from transformers import Trainer
from kairos.integrations import KairosHFCallback

trainer = Trainer(
    model=model, args=args, ...,
    callbacks=[KairosHFCallback(bundle, save_dir="checkpoints/")],
)
trainer.train()
```

### PyTorch Lightning

```python
import pytorch_lightning as pl
from kairos.integrations import KairosLightningCallback

trainer = pl.Trainer(
    max_steps=n_steps,
    callbacks=[KairosLightningCallback(bundle, save_dir="checkpoints/")],
)
trainer.fit(model, dataloader)
```

## Hyperparameter sweeps

```python
from kairos import KairosSweepGate

gate = KairosSweepGate(n_trials=8, eval_at_step=2000, keep_top_k=2)
for step in range(...):
    for trial_id, trial in trials.items():
        metrics = train_one_step(trial)
        gate.observe_trial(trial_id, step, **metrics)
    if step == 2000:
        for trial_id, dec in gate.make_decisions().items():
            if dec.kill:
                trials[trial_id].alive = False
                print(f"killed {trial_id}: {dec.reason}")
```

## Probing capability emergence (research)

```python
from kairos import KairosProbe

probe = KairosProbe()
for step in eval_steps:
    scores = {"icl": eval_icl(model), "cot": eval_cot(model),
              "multi_hop": eval_multi_hop(model)}
    probe.observe(step, scores=scores)

for report in probe.diagnose():
    if report.likely_to_emerge_next:
        print(f"{report.capability} approaching emergence "
              f"(regime={report.regime}, AR(1) τ={report.ar1_trend_tau:+.2f})")
```

## Phase classification

Each `observe()` returns a `Phase`:

| Phase | Meaning |
|---|---|
| `MEMORISING` | train_acc < memorisation_threshold |
| `PLATEAU` | train_acc ≈ 1, test_acc near chance |
| `NEAR_CRITICAL` | Cassandra CSD signature; transition imminent |
| `DRIFTING` | Cassandra signals movement; test metrics changing |
| `POST` | Grokking event detected; test_acc high |

## Real-training validation

See `examples/train_with_kairos.py` for an end-to-end head-to-head:
baseline AdamW Transformer vs. Kairos-wrapped version, same seed,
same hparams. Demonstrates early-stop on memorisation-only runs,
LR drop at transition, and accelerator pulses during the plateau.

## Tests

```bash
pip install -e .
python -m pytest tests/ -q          # 23 tests, < 10s
```

## License

MIT.
