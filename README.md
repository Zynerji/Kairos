# Kairos

> *καιρός* — Greek: "the right moment, the critical time"

**Grokking-aware training-optimizer toolkit.** Built on
[Grokking-Monitor](https://github.com/Zynerji/Grokking-Monitor) and
[Cassandra](https://github.com/Zynerji/Cassandra). Detects the
moment a network transitions from memorisation to generalisation and
acts on it.

## Components — what ships vs what's research

| # | Component | Use today? | Why |
|---|---|---|---|
| 1 | `KairosEarlyStop`    | ✅ ship       | Clear compute savings on dead-end runs. Validated on GPU. |
| 2 | `KairosLRSchedule`   | ⚠️ careful    | Gates on `monitor.detected_event`, not raw Cassandra regime. Helps on sharp groks; no-op on slow groks. |
| 3 | `KairosCheckpoint`   | ✅ ship       | Gates on `monitor.detected_event`. Captures the right state on sharp groks. |
| 4 | `KairosSweepGate`    | ✅ ship       | Multi-trial allocator. Validated to rank grokking trial above memorising trial. |
| 5 | `KairosAccelerator`  | 🔬 research   | Weight-noise pulses during plateau. Stops firing when Cassandra signals movement — to prevent disrupting an emerging solution. |
| 6 | `KairosCurriculum`   | 🔬 research   | Phase-aware LR/WD. Includes hysteresis + ratchet to handle Cassandra regime flapping on slow-grok runs. |
| 7 | `KairosProbe`        | 🔬 research   | Tracks multiple capability scores in parallel; flags emergence precursors. |

## Empirical results (RTX PRO 4000 Blackwell, p=29 modular arithmetic, AdamW lr=1e-3 wd=1.0)

### EarlyStop saves real compute on dead-end runs

| | wall clock | steps run | final test_acc |
|---|---|---|---|
| BASELINE_DEAD (wd=0.0) | 246.8s | 15,000 | 0.202 |
| EARLY_STOP_DEAD       | **87.9s** | **10,066** (aborted) | 0.190 |

**64% compute saved.** EarlyStop fired at step 10,065 after 10,000
post-memorisation steps with no grokking signature. Same final test
accuracy within noise — confirming the run was not going to grok.

Reproduce: `python examples/early_stop_demo.py`.

### Slow-grok regime (15K-step head-to-head)

| | wall clock | final test_acc |
|---|---|---|
| BASELINE | 211.1s | **0.365** |
| KAIROS-C (EarlyStop + LRSchedule + Checkpoint) | 236.3s | **0.365** ✓ |
| KAIROS-F (full stack incl. Accelerator + Curriculum) | 217.2s | 0.168 ✗ |

`KAIROS-C` matches baseline exactly — correct behaviour on a run
that doesn't grok within budget (LRSchedule + Checkpoint gate on
`monitor.detected_event`, which never fires here). The +12% wall
overhead is from Cassandra `diagnose()` calls every 200 steps.

`KAIROS-F` still hurts on this slow-grok regime because the
accelerator's weight-noise pulses disrupt the slowly-emerging
solution. **This is why Accelerator + Curriculum are marked
research; they need a sharper-grok regime than this to demonstrate
benefit.** Reproducing Power-2022's sharp grok on p=97 with
high-quality hparams is on the roadmap.

### Validation against synthetic curves (CPU)

23 tests pass on local + VM. Synthetic-curve tests verify that the
monitor fires on synthetic sharp groks, that EarlyStop doesn't fire
when grokking, that SweepGate ranks a grokking trial above a
memorising one, etc.

## Quick start

```python
from kairos import (
    CallbackBundle, KairosEarlyStop, KairosLRSchedule,
    KairosCheckpoint, KairosCurriculum, KairosAccelerator,
)

bundle = CallbackBundle(
    KairosEarlyStop(stable_steps_to_abort=10_000, min_step=2000),
    KairosLRSchedule(drop_factor=0.1, optimizer=opt),
    KairosCheckpoint(save_dir="checkpoints/"),
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

## Framework integrations

* `from kairos.integrations import KairosHFCallback`        # HuggingFace Trainer
* `from kairos.integrations import KairosLightningCallback`  # PyTorch Lightning

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
```

## Phase classification

`Action.phase` is one of: `MEMORISING`, `PLATEAU`, `NEAR_CRITICAL`,
`DRIFTING`, `POST`, `UNKNOWN`. `KairosCurriculum`'s ratchet means the
phase never regresses below the deepest one observed.

## Tests + demos

```bash
pip install -e .
python -m pytest tests/ -q                            # 23 tests
python examples/early_stop_demo.py                    # the demonstrated win
python examples/train_with_kairos.py --mode all       # baseline / kairos-c / kairos-f
python examples/sweep_demo.py                         # KairosSweepGate ranking
```

## License

MIT.
