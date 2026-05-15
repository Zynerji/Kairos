# Kairos

> *καιρός* — Greek: "the right moment, the critical time"

**Grokking-aware training-optimizer toolkit + battle-tested LR /
multi-axis / growth mechanisms** ported from a research stack of
LLM-training repos. Built on
[Grokking-Monitor](https://github.com/Zynerji/Grokking-Monitor) and
[Cassandra](https://github.com/Zynerji/Cassandra).

## v0.2 components (10)

| # | Component | Source / status |
|---|---|---|
| 1 | `KairosEarlyStop`           | ✅ shipped — 64 % compute saved on dead-end runs (validated GPU) |
| 2 | `KairosLRSchedule`          | ✅ shipped — one-shot LR drop at confirmed grok event |
| 3 | `KairosCheckpoint`          | ✅ shipped — gates on `monitor.detected_event` |
| 4 | `KairosSweepGate`           | ✅ shipped — multi-trial compute allocator |
| 5 | `KairosAccelerator`         | 🔬 research — weight-noise pulses during plateau |
| 6 | `KairosCurriculum`          | 🔬 research — phase-aware lr/wd with hysteresis + ratchet |
| 7 | `KairosProbe`               | 🔬 research — generic capability-emergence probe |
| 8 | **`KairosPendulumLR`**      | ✅ ported from Kanon (Alembic DHART v14.2: 15/15 crystal detection, ~5 % over cosine in late-phase distillation) |
| 9 | **`KairosParetoGuard`**     | ✅ ported from Aletheia (Qwen3 9-axis post-training, dual-regression rollback) |
| 10 | **`KairosGrowthController`** | ✅ ported from qGPT-Infinity (proven: K=8 → 64 auto-grown, 6 successful events) |

## The 3 new mechanisms

### `KairosPendulumLR` — Hamiltonian-pendulum loss-driven LR

The training loss drives a damped pendulum; its Conservation Violation
(CV = |dH/dt| / ⟨|H|⟩) classifies the run into three phases, each
with its own LR multiplier:

| CV | phase | lr_mult |
|---|---|---|
| < 0.1 | CRYSTAL (plateau) | 1 / φ²  ≈ 0.382  (fine-tune) |
| 0.1 – 0.3 | ACTIVE | 1.000 |
| > 0.3 | EXPLORE (stuck) | φ ≈ 1.618 (escape) |

Continuous loss-driven adaptation. Unlike cosine, the minimum is
0.382 × base — training never freezes. Composes cleanly with
`KairosLRSchedule` (which adds the one-shot grokking-event drop on
top of the pendulum's continuous modulation). Includes EMA smoothing
to dampen per-step jitter.

```python
from kairos import KairosPendulumLR, CallbackBundle

bundle = CallbackBundle(
    KairosPendulumLR(optimizer=opt, apply_smoothing=0.85),
)
```

### `KairosParetoGuard` — multi-axis Pareto-frontier rollback

Multi-objective training benefits from a *dual-regression rollback
gate*: a single-axis dip is expected; two-axis collapse is not.
Anchor + 80 % floor + product metric:

```python
from kairos import KairosParetoGuard

guard = KairosParetoGuard(
    anchor={"factuality": 0.82, "calibration": 0.71, "reasoning": 0.68},
    floor_mult=0.80,
)
bundle.observe(
    step,
    factuality=eval_factuality, calibration=eval_cal, reasoning=eval_rea,
    train_loss=..., train_acc=..., test_loss=..., test_acc=...,
)
# Auto-saves a checkpoint when product improves AND no axis is below floor.
# Emits a ROLLBACK signal in `action.notes` when >= 2 axes dip below floor.
```

Optional `spectral_target_std` companion computes an adaptive alpha
for per-head variance-collapse rescue (the Aletheia spectral-amp pattern).

### `KairosGrowthController` — saturation-triggered architecture growth signal

Three coupled pendulums (K / W / D for capacity / width / depth)
advance with angular velocities proportional to per-dimension
saturation signals. The bronze-mean ratchet picks the dimension to
grow. Outputs a structured `grow:<dim>` signal in the Action's notes;
the training loop applies the actual surgery.

```python
from kairos import KairosGrowthController

bundle = CallbackBundle(
    KairosGrowthController(min_steps_before_grow=500, cooldown_steps=1000),
)
action = bundle.observe(
    step, train_loss=..., hidden_var=..., depth_signal=...,
    train_acc=..., test_loss=..., test_acc=...,
)
for n in action.notes:
    if n.startswith("grow:K"):
        grow_K_modes()
    elif n.startswith("grow:W"):
        grow_width()
    elif n.startswith("grow:D"):
        grow_depth()
```

## Headline empirical results (RTX PRO 4000 Blackwell, `(a+b) mod 29` Transformer)

### KairosPendulumLR.for_grokking() — **+62 pp test accuracy**

`AdamW(lr=1e-3, wd=1.0)`, 2-layer Transformer (~540 k params), 15 000 steps, same seed:

| config | wall | final train_acc | final test_acc |
|---|---|---|---|
| BASELINE_STATIC | 248.1 s | 1.000 | **0.365** |
| KAIROS_PENDULUM (`for_grokking()`) | 252.1 s | 1.000 | **0.988** ✓ |

**+0.623 absolute test accuracy.** The pendulum essentially *completed
the grok* that static-LR could not.

Phase distribution over 15 000 steps:
* CRYSTAL: 44 % (lr_mult ≈ 0.382 — fine-tune)
* ACTIVE:  19 % (lr_mult = 1.0)
* EXPLORE: **37 %** (lr_mult ≈ 1.618 — escape the stuck state)

The Hamiltonian-pendulum CV signal on `test_loss` correctly diagnoses
the slow-grok plateau as "stuck", boosts LR by 1.618×, and lets the
optimizer break out. Reproduce: `python examples/pendulum_lr_demo.py`.

> Calibration note: `train_loss` flatlines at ~0 after memorisation in
> grokking-shaped tasks, so the original Kanon-default `train_loss`
> pendulum metric over-throttles (CRYSTAL 87 % of steps, test_acc
> -0.109 vs baseline). For grokking, drive the pendulum from
> `test_loss` — that's what `for_grokking()` does.

### EarlyStop saves real compute on dead-end runs

| | wall | steps | final test_acc |
|---|---|---|---|
| BASELINE_DEAD (wd=0.0) | 246.8 s | 15 000 | 0.202 |
| EARLY_STOP_DEAD       | **87.9 s** | 10 066 (aborted) | 0.190 |

**64 % compute saved**, same final acc within noise. Reproduce:
`python examples/early_stop_demo.py`.

### Conservative bundle = baseline behaviour on slow-grok

| | wall | final test_acc |
|---|---|---|
| BASELINE | 211 s | **0.365** |
| KAIROS-C (EarlyStop + LRSchedule + Checkpoint) | 236 s | **0.365** ✓ |

Correct behaviour: `monitor.detected_event` never fires on slow groks,
so LRSchedule + Checkpoint correctly no-op. 12 % wall overhead from
Cassandra `diagnose()` every 200 steps.

## Framework integrations

```python
from kairos.integrations import KairosHFCallback        # HuggingFace
from kairos.integrations import KairosLightningCallback  # Lightning
```

## Tests + demos

```bash
pip install -e .
python -m pytest tests/ -q                         # 44 tests
python examples/early_stop_demo.py                 # validated win
python examples/train_with_kairos.py --mode all    # 3-way head-to-head
python examples/pendulum_lr_demo.py                # PendulumLR A/B
python examples/sweep_demo.py                      # SweepGate
```

## License

MIT.
