# Changelog

## 0.5.1 — 2026-05-15

**Performance + scope fixes for `WeightDeltaCodebook`:**

- SVD now runs on GPU when available (default: autodetect CUDA, override
  with `svd_device=`). CPU SVD on 1536 × 6144 weight deltas took minutes
  per layer; GPU does it in milliseconds. Codebook build over 50 layers
  drops from ~hours to ~seconds.
- `apply_restoration` no longer rejects `alpha > 1.0`. Over-injection is
  useful for diagnosing whether capability and refusal directions are
  coupled — if α = 4 still doesn't move metrics, the directions overlap.
- `examples/aletheia_codebook_validate_v2.py` — proper validation
  scaffold with capability prompts pulled from real benchmarks (GSM8K,
  TriviaQA, alpaca, OpenThoughts), 64 prompts per axis, 4 axes.

**Validation findings on (google/gemma-4-E2B-it,
huihui-ai/Huihui-gemma-4-E2B-it-abliterated):**

| metric | A | B | C(α=1) | C(α=2) | C(α=4) | C(α=8) |
|---|---|---|---|---|---|---|
| GSM8K accuracy   | 0.0333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| refusal rate     | 0.9333 | 0.0333 | 0.0333 | 0.0333 | 0.0667 | 0.0667 |
| per-layer alpha_cap | — | — | **0.077** uniform across all 50 paired layers |

The result is **inconclusive** for this specific pair: huihui-ai's
abliteration of Gemma 4 E2B is too gentle on math/factuality/instruction/
reasoning to demonstrate codebook healing at our 30-sample eval
resolution. The capability fraction of every per-layer ΔW is uniformly
~7.7%, so even 8× over-injection only re-introduces ~62% of the rank-1
update — and that small fraction doesn't move the eval needle.

The mechanism is mathematically correct (29 unit tests pass, math
verified on synthetic data). The demonstration target was wrong. Need
either (a) larger eval samples (200–500 to resolve sub-30 effects), or
(b) a more aggressively abliterated checkpoint pair.

Tests: 199/199 passing (was 198).

## 0.5.0 — 2026-05-15

**`kairos.aletheia.surgery` — refusal-direction abliteration + healing
primitives.**

Two complementary tools for working with refusal-direction abliteration
(Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction", 2024). Standard abliteration projects out a raw refusal
direction `r` from every weight matrix that writes to the residual
stream, which damages any capability that co-fires with refusal.
This module gives you two ways to do it better:

**Path A — `CapabilityAwareAbliterator`** (in-line at abliteration time)

  - Compute the refusal direction `r` and a capability subspace `C`
    via diff-of-means on harmful / harmless / per-axis-capability
    prompts.
  - Orthogonalise: `r_pure = r − C·(Cᵀ·r)`.
  - Project `r_pure` (not `r`) out of every target weight matrix.

  Result: refusal removed, capabilities correlated with the standard
  refusal direction are preserved. Reports per-axis overlap so you
  can tell which capabilities the standard recipe would have damaged.

**Path B — `WeightDeltaCodebook`** (post-hoc on already-abliterated models)

  - Given un-abliterated + abliterated state dict pair, build per-layer
    `ΔW_ℓ = W_original_ℓ − W_abliterated_ℓ` (≈ rank-1 by construction).
  - SVD-decompose. Split the rank-1 direction into capability-aligned
    and capability-orthogonal components via the same `C` subspace.
  - Selective re-injection at `α ∈ [0, 1]`:
    `W_healed = W_abliterated + α · ΔW_capability`.
  - α-sweep + per-axis OOT eval picks the Pareto-best operating point.

**Primitives**

  - `compute_direction_from_activations(harmful, harmless)` — diff-of-means
  - `compute_capability_subspace(axes, neutral)` — orthonormal basis from
    per-axis diff-of-means
  - `project_out_subspace(direction, subspace)` — Gram-Schmidt remove

**Examples**

  - `examples/aletheia_capability_aware_abliterate.py` — full Path A
    pipeline. Probes residual-stream activations on a 50-prompt corpus,
    builds refusal + capability, runs the abliterator, saves codebook.
  - `examples/aletheia_codebook_restore.py` — full Path B pipeline.
    Loads un-abliterated + abliterated pair, builds the codebook,
    α-sweeps with `kairos.aletheia.pools.*` evaluators.

**Tests**

  - `tests/aletheia/surgery/` — 28 tests for the math (math correctness
    of refusal direction, capability subspace orthonormality, codebook
    rank-1 recovery, alpha-zero/one identities, in-place mutation,
    skip-substring filter, etc.).

  Total now: **198 tests passing** (was 170).

Note: the placeholder prompt corpora in the examples (`HARMFUL_PROMPTS`,
`HARMLESS_PROMPTS`, `CAPABILITY_PROMPTS`) are illustrative — 8 prompts
per class. Real probing for a publishable abliteration needs proper
corpora (HarmBench / AdvBench for harmful, paired same-distribution
harmless, 100+ per capability axis drawn from MMLU/GSM8K/TriviaQA). The
math is correct at any corpus size; only the resulting direction
quality varies.

## 0.4.0 — 2026-05-15

**Aletheia salvage — major subpackage added.**

The Aletheia repo (post-training stack for ablated/uncensored LLMs)
was retired on 2026-05-15. Its substantive contents have been
imported into Kairos as the `kairos.aletheia` subpackage. Aletheia
will not be developed further as a standalone codebase; all future
work happens here.

What was salvaged:

- `kairos/aletheia/torsion/` — bronze pendulum, **torus T² pendulum**
  (`(2π/φ², 2π/β₃)` quasiperiodic schedule), spectral amplification,
  Phase A/B cycle controller. The bronze pendulum overlaps in spirit
  with the existing `KairosPendulumLR` but the torus T² and Phase A/B
  cycle are new capabilities Kairos didn't have.
- `kairos/aletheia/ratchet/` — the original Aletheia Pareto ratchet
  source that `KairosParetoGuard` was ported from. Kept in case the
  original API is preferred for direct use.
- `kairos/aletheia/pools/` — 9 named training pools (factuality,
  calibration, abstention, grounding, consistency, sycophancy,
  reasoning, instruction, distillation) on top of `CausalLMPool` /
  `HFCausalLMPool` bases.
- `kairos/aletheia/adapters/lora_per_pool.py` — multi-named-adapter
  LoRA registration for Phase A per-pool cycling.
- `kairos/aletheia/distill/` — refusal `teacher_filter` (regex +
  classifier) and `rejection_sample` (re-roll on filter failure).
- `kairos/aletheia/eval/` — `held_out` per-pool OOT gate and
  `benchmarks` task-shape metric harness (the missing piece from
  yesterday's perplexity-axis caveat).
- `kairos/aletheia/growth/` — confidence-head / pool-side-FFN /
  expert-addition stubs. Off by default.
- `kairos/aletheia/phase_b.py` — combined Phase B loss.
- `configs/aletheia/` — 5 original yaml configs (pendulums, ratchet,
  growth, base_dev, base_prod).
- `examples/aletheia/` — 4 original scripts (train_dev_8b,
  train_prod_42b, baseline_eval, prepare_teacher_corpus).

Plus 98 new tests under `tests/aletheia/`. Total: **170 tests
passing** (was 72).

**Fixed in salvage:** Aletheia's `tests/test_hf_helpers.py::
test_partial_overlap` had a bug — the test used `"a b c"` as a pred,
but `_normalize_text` strips articles (`a/an/the`), so the actual
F1 is 0.8 (precision=1.0, recall=2/3), not 2/3. Test rewritten to
use non-article tokens.

The `examples/aletheia_gemma4_heal.py` scaffold from yesterday is
still in `examples/` as the integration demo of `KairosParetoGuard`
+ `kairos.aletheia.pools` would slot in cleanly to replace its
perplexity axes with task-shape metrics from
`kairos.aletheia.eval.held_out`.

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
