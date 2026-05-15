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

## Honest empirical findings (modular arithmetic on RTX PRO 4000 Blackwell)

We ran head-to-heads on `(a + b) mod 29` with `AdamW(lr=1e-3, wd=1.0)`,
2-layer Transformer (~ 540k params), 15K steps.

**Slow-grok regime** (the canonical CPU/small-model case):
* Train accuracy reaches 1.0 by step ~150
* Test accuracy creeps from 0.05 toward ~ 0.4 over 15K steps
* No sharp transition; monitor never fires `detected_event`
* Therefore LRSchedule + Checkpoint are no-ops here — exactly the right behaviour
* Conservative bundle (EarlyStop + LRSchedule + Checkpoint) behaves like baseline

**Active steering on slow-grok runs HURT in our v0.1.0 prerelease config.**
The pre-release `KairosLRSchedule` dropped LR on raw `drifting` regime
(which fires thousands of steps before grokking on slow groks);
dropped from `0.365` baseline to `0.154` test_acc. We fixed it: now
gates on the confirmed event. Same root-cause issue we found and
documented for `KairosAccelerator` and `KairosCurriculum`; both got
hysteresis / ratchet semantics in response.

**Clear win**: `KairosEarlyStop` on dead-end runs (no grok signature
within budget) demonstrably saves compute. See
`examples/early_stop_demo.py` for the validated numbers.

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
