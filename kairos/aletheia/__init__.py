"""kairos.aletheia — salvaged Aletheia stack (post-training for ablated LLMs).

Imported into Kairos from the deprecated `Aletheia/` repo on 2026-05-15
(commit-rewrite of Kairos v0.4.0). Original Aletheia code structure
preserved verbatim except for namespace rewrite `aletheia` →
`kairos.aletheia`.

What this gives you on top of plain Kairos:

  * `torsion/`     — bronze pendulum, torus T² pendulum, spectral
                     amplification, Phase A/B cycle controller
  * `ratchet/`     — the original Aletheia Pareto ratchet (the source
                     `KairosParetoGuard` was ported from)
  * `pools/`       — 9 named training pools (factuality, calibration,
                     abstention, grounding, consistency, sycophancy,
                     reasoning, instruction, distillation) on top of
                     `CausalLMPool` / `HFCausalLMPool` bases
  * `adapters/`    — `lora_per_pool`: per-pool LoRA adapter registration
                     for Phase A cycling
  * `distill/`     — `teacher_filter` (refusal regex + classifier) and
                     `rejection_sample` (re-roll teacher generations on
                     filter failure)
  * `eval/`        — `held_out` (per-pool OOT gate) + `benchmarks`
  * `growth/`      — confidence-head / pool-side-FFN / expert-addition
                     **STUBS** (off by default; do NOT enable without
                     explicit growth plan)
  * `phase_b.py`   — combined Phase B loss (distill_ce + distill_kl + Brier)

The Aletheia targets carry over: dev = `mlabonne/Qwen3-8B-abliterated`,
prod = the DavidAU Qwen3-42B abliterated thinker. `examples/aletheia/`
holds the original scripts (`train_dev_8b.py`, `train_prod_42b.py`,
`baseline_eval.py`, `prepare_teacher_corpus.py`) and `configs/aletheia/`
holds the original 5 yaml configs.
"""

__version__ = "0.4.0"
