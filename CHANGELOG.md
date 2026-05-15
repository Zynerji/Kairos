# Changelog

## 0.3.1 — 2026-05-15

**HF Trainer integration — first real-model validation**

- `KairosHFCallback` now subclasses `transformers.TrainerCallback`
  (was duck-typed). HF Trainer 5.x is strict about the callback
  surface — calls `getattr(callback, event)` for every event
  including `on_init_end`, which the duck-typed version didn't
  implement. The subclassing is conditional on `transformers` being
  importable, so the existing mock-based tests still pass (6/6).

- `examples/gemma4_smoke.py` — single-forward load test for any HF
  multimodal causal-LM (used to validate Gemma 4 E2B abliterated on a
  24 GB Blackwell: 5.104B params total, 10.21 GB VRAM after load,
  forward at 10.28 GB peak).

- `examples/gemma4_lora_sft.py` — LoRA SFT scaffold for Gemma 4
  multimodal models with three Gemma-4-specific landmines documented:
  1. PEFT must target the text decoder's projections *only*. Gemma 4
     wraps vision + audio encoder projections in `Gemma4ClippableLinear`
     (`torch.clamp(x, -inf, +inf)` then `nn.Linear`); those paths are
     dormant on text-only inputs. Targeting them produces
     `grad_norm=0` and flat loss. Use a regex anchored on
     `language_model.layers.\d+.(self_attn.(q|k|v|o)_proj|mlp.(gate|up|down)_proj)$`.
  2. `model.enable_input_require_grads()` is required after
     `get_peft_model()` — otherwise the embedding output has
     `requires_grad=False` and gradients don't reach LoRA matrices.
  3. Use `TrainingArguments(lr_scheduler_type="constant", warmup_steps=0)`
     so `KairosPendulumLR` owns the LR (HF's built-in scheduler
     would compete with the pendulum's modulation).

**Validation (Gemma 4 E2B abliterated, RTX PRO 4000 Blackwell)**

- 50 grad updates × batch=1 × grad_accum=8 × seq=1024 = 410k tokens
  in 3:18 wall, VRAM peak 20.79 GB
- LoRA: 12.08M trainable params (rank=8, target text decoder only)
- Train loss: **5.09 → 1.37** (perplexity 162 → 4)
- `grad_norm`: 7.33 → 0.58 (healthy convergence, was 0 with broken
  targeting)
- Generation post-SFT cleanly adopts the R1-distill "Step-by-step
  thinking process" format with the markdown step list pattern
  characteristic of OpenThoughts traces.

## 0.3.0 — 2026-05-15

**New components**

- `KairosAntiResonantInit` (component #11): orthogonal weight init that
  projects out a teacher's top-K singular directions. Ported from
  qGPT-Infinity (`core/dirac_crystal_linear.py`, fixed K=8 silver-init
  NaN). Also handles `nn.Embedding` with the phase-staggered Fourier
  trick. Verified: student-teacher top-K overlap < 0.15 (vs 0.25
  random isotropic baseline) when `suppress_top_k=4`.

- `recommended_bundle(profile, **kwargs)`: one-line factory for the
  five canonical training profiles — `"grokking"`, `"distillation"`,
  `"pareto_post_training"`, `"growth_search"`, `"pretraining"`.

- `KairosPendulumLR.for_grokking()` / `.for_distillation()` presets
  (formalised in v0.2; documented as multi-seed validated in v0.3).

- Mamba scaffolds: `examples/train_mamba_1b.py` (1.3B pretrain) and
  `examples/finetune_deepseek_r1.py` (SFT on R1-Distill thinking
  traces with refusal filtering).

**Test coverage**

- `tests/test_v03_components.py` — 16 tests for antiresonant init +
  recommended_bundle.
- `tests/test_integrations_mocked.py` — 6 smoke tests for the
  HF/Lightning adapters using mock trainers (no framework install
  required).
- Total: **68 tests passing** (was 46).

**Headline validation (5-seed paired study)**

- `KairosPendulumLR.for_grokking()` vs static-LR baseline on `(a+b) mod 29`
  modular-arithmetic Transformer, RTX PRO 4000 Blackwell, 15 000 steps
  per run:
  - Seed 0: 0.365 → 0.988 (+0.623) — pendulum escaped slow-grok plateau
  - Seed 1: 0.399 → 0.766 (+0.367) — pendulum escaped slow-grok plateau
  - Seed 2: 0.864 → 1.000 (+0.136) — both groked, pendulum finished cleaner
  - Seed 3: 0.990 → 0.990 (+0.000) — both already groked, pendulum no-op
  - Seed 4: 0.251 → 0.316 (+0.065) — neither groked, pendulum still ahead
  - **Mean delta +0.238 ± 0.256, pendulum > baseline at 4/5 seeds.**
  - Key finding: pendulum never hurts, escapes stuck-grok in 2/5 cases.

**0.74B Mamba pretraining demonstrator**

- `examples/train_mamba_1b.py` end-to-end on Blackwell:
  - 737.6 M params (Mamba d_model=2048, n_layer=24, vocab 50 280)
  - C4 streaming via HuggingFace datasets, GPT-NeoX tokeniser
  - `recommended_bundle("pretraining")` drives the LR
  - 1 000 grad updates (~8.2 M tokens) in 43.5 min
  - **Loss 8.31 → 5.42 (perplexity 4 083 → 227)**
  - Generation: grammatical English, no semantic coherence (expected
    at this token count vs ~300 B for canonical Mamba-790m)

**Bug fix during launch**

- `KairosPendulumLR.set_initial_lrs(lr)` — re-anchor the pendulum's
  base LR after warmup (the captured-on-first-observe LR was the
  warmup LR, anchoring the pendulum 100× below target).

**DeepSeek-R1 fine-tune demonstrator**

- `examples/finetune_deepseek_r1.py` end-to-end on the pretrained
  0.74 B ckpt:
  - `open-thoughts/OpenThoughts-114k` (sharegpt-style `conversations`
    schema; refusal-filtered)
  - `recommended_bundle("distillation")` LR pendulum, anchor 5e-5
  - 300 grad updates (~4.9 M tokens) in 26.7 min
  - **Loss 6.40 → 2.95 (perplexity 603 → 19)**
  - Model cleanly emits `<|begin_of_thought|>` + R1-distill thinking
    style on every prompt within 30 grad-updates (~2.5 min)
- Sharegpt `conversations` schema added to `format_example` (handles
  OpenThoughts, Bespoke-Stratos, s1K with `{from: user/gpt, value: ...}`
  rows).
- Checkpoint rotation (`--keep-last`) added; disk-fill no longer
  crashes the run mid-save.

**Canonical 1.37B Mamba with 8-bit Adam**

- `--canonical-1p4b --adam-8bit` flags wired into
  `examples/train_mamba_1b.py`. Verified VRAM footprint:
  **14.05 GB peak** at batch=1 seq=1024 (vs OOM at fp32 AdamW on the
  same 24 GB GPU). 1.37 B params, n_layer=48, d_model=2048.
- `examples/launch_mamba_pretraining.sh` accepts `CONFIG=1p4b` to opt
  in to the 1.4B + 8-bit Adam configuration.

## 0.2.0 — 2026-05-14

**New components**

- `KairosPendulumLR` (component #8): Hamiltonian-pendulum loss-driven
  LR adaptation. Ported from Kanon `src/kanon/training/pendulum.py`,
  proven in Alembic DHART v14.2.
- `KairosParetoGuard` (component #9): multi-axis Pareto-frontier
  rollback gate. Ported from Aletheia; Qwen3 9-axis post-training,
  dual-regression rollback.
- `KairosGrowthController` (component #10): saturation-triggered
  architecture-growth signal. Ported from qGPT-Infinity (K=8 → 64
  auto-grown, 6 successful events).

**Headline result**

- `KairosPendulumLR.for_grokking()`: **+62 pp** absolute test accuracy
  on a modular-arithmetic grok run (single-seed GPU result).

## 0.1.0 — 2026-05-13

Initial release: 7 grokking-aware components (EarlyStop, LRSchedule,
Checkpoint, SweepGate, Accelerator, Curriculum, Probe) on top of
Grokking-Monitor + Cassandra.

- EarlyStop validation: 64% compute saved on dead-end runs.
