# Kairos

> *καιρός* — Greek: "the right moment, the critical time"

**Grokking-aware training-optimizer toolkit + battle-tested LR /
multi-axis / growth mechanisms** ported from a research stack of
LLM-training repos. Built on
[Grokking-Monitor](https://github.com/Zynerji/Grokking-Monitor) and
[Cassandra](https://github.com/Zynerji/Cassandra).

## v0.3 components (11) + one-line factory

| # | Component | Source / status |
|---|---|---|
| 1 | `KairosEarlyStop`           | ✅ shipped — 64 % compute saved on dead-end runs (validated GPU) |
| 2 | `KairosLRSchedule`          | ✅ shipped — one-shot LR drop at confirmed grok event |
| 3 | `KairosCheckpoint`          | ✅ shipped — gates on `monitor.detected_event` |
| 4 | `KairosSweepGate`           | ✅ shipped — multi-trial compute allocator |
| 5 | `KairosAccelerator`         | 🔬 research — weight-noise pulses during plateau |
| 6 | `KairosCurriculum`          | 🔬 research — phase-aware lr/wd with hysteresis + ratchet |
| 7 | `KairosProbe`               | 🔬 research — generic capability-emergence probe |
| 8 | **`KairosPendulumLR`**      | ✅ ported from Kanon — 5-seed: wins 4/5 (+0.24 mean, +0.62 best) on modular-arithmetic grok (RTX PRO 4000) |
| 9 | **`KairosParetoGuard`**     | ✅ ported from Aletheia (Qwen3 9-axis post-training, dual-regression rollback) |
| 10 | **`KairosGrowthController`** | ✅ ported from qGPT-Infinity (K=8 → 64 auto-grown, 6 successful events) |
| 11 | **`KairosAntiResonantInit`** | ✅ ported from qGPT-Infinity — orthogonal init avoiding teacher harmonics (fixed silver-init NaN at K=8 distill) |

### One-line factory

```python
from kairos import recommended_bundle

bundle = recommended_bundle("grokking", optimizer=opt,
                             max_steps=15000,
                             checkpoint_dir="./ckpt")
# Profiles: "grokking" | "distillation" | "pareto_post_training"
#           | "growth_search" | "pretraining"
```

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

### KairosPendulumLR.for_grokking() — **5-seed paired study**

`AdamW(lr=1e-3, wd=1.0)`, 2-layer Transformer (~540 k params), 15 000
steps, 5 seeds (0-4), same data split per seed across BASELINE_STATIC
and KAIROS_PENDULUM (`for_grokking()`):

| config | mean final test_acc | std | per-seed |
|---|---|---|---|
| BASELINE_STATIC                       | 0.574 | 0.330 | 0.365 / 0.399 / 0.864 / 0.990 / 0.251 |
| KAIROS_PENDULUM (`for_grokking()`)    | **0.812** | 0.294 | 0.988 / 0.766 / 1.000 / 0.990 / 0.316 |

**Paired delta (PENDULUM − BASELINE):** mean **+0.238 ± 0.256**,
pendulum > baseline at **4 / 5 seeds** (delta > +0.05). Per-seed:
`[+0.623, +0.367, +0.136, +0.000, +0.065]`.

Interpretation by regime:

* **Baseline gets stuck (seeds 0, 1):** pendulum escapes the
  memorisation plateau and groks. Single-seed headline: +0.623 / +0.367.
* **Baseline groks within 15 k steps (seeds 2, 3):** pendulum matches
  it (0.990–1.000). No regression from adding the pendulum.
* **Neither config groks in 15 k steps (seed 4):** pendulum still
  ahead (0.316 vs 0.251) — both need more steps.

The Hamiltonian-pendulum CV signal on `test_loss` diagnoses the slow-grok
plateau as "stuck", boosts LR by φ ≈ 1.618×, and lets the optimizer
break out. Reproduce: `python examples/multi_seed_validation.py`.

> Calibration note: `train_loss` flatlines at ~0 after memorisation in
> grokking-shaped tasks, so the original Kanon-default `train_loss`
> pendulum metric over-throttles. For grokking, drive the pendulum
> from `test_loss` — that's what `for_grokking()` does.

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
python -m pytest tests/ -q                         # 72 tests
python examples/early_stop_demo.py                 # validated win
python examples/train_with_kairos.py --mode all    # 3-way head-to-head
python examples/pendulum_lr_demo.py                # PendulumLR A/B
python examples/multi_seed_validation.py           # 5-seed GPU robustness
python examples/sweep_demo.py                      # SweepGate
python examples/train_mamba_1b.py --smoke          # 0.74B Mamba scaffold (smoke)
python examples/finetune_deepseek_r1.py --smoke    # DeepSeek-R1 SFT scaffold (smoke)
# Real launches on Blackwell:
bash   examples/launch_mamba_pretraining.sh                # 0.74B Mamba, fp32 Adam
CONFIG=1p4b bash examples/launch_mamba_pretraining.sh      # 1.37B Mamba, 8-bit Adam
```

See `results/SUMMARY.md` for the headline runs (5-seed multi-seed
validation, 0.74B pretrain → DeepSeek SFT, +0.74B/+1.37B VRAM
benchmarks).

## License

MIT.
