"""Aletheia-style multi-axis Pareto LoRA heal for an abliterated LLM.

v2 (2026-05-15): swapped the ``exp(-loss)`` perplexity axes for real
task-shape metrics via ``kairos.aletheia.pools.*``:

    - ReasoningPool:    GSM8K train + GSM8K test eval, number-extract accuracy
    - FactualityPool:   TriviaQA (`rc.nocontext`) train + held-out eval, F1
    - InstructionPool:  Tulu-3 train + IFEval eval, F1 proxy

The earlier ``exp(-loss)`` axes (v1) measured perplexity on held-out
text that included our chat-template tokens — a metric the model
trivially improves on just by absorbing the format, not by gaining
capability. Now we use the same evaluation harness Aletheia was
designed for: greedy generation, decode, score against gold via
task-appropriate metrics.

Run on the VM
=============
    PYTHONPATH=. python3 examples/aletheia_gemma4_heal.py \\
        --max-steps 100 --batch-size 1 --grad-accum 8 \\
        --eval-every 25 --eval-samples 16 --train-size-per-pool 256

Per-eval cost
=============
Each periodic OOT eval generates ``eval_samples`` × max_new_tokens
tokens per pool (3 pools, default 16 samples × ~128 tokens). On the
Blackwell at ~50 tok/s for a 5B multimodal model that's roughly
2-3 min per eval pass. Eval freq is the main scaling knob.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import (  # noqa: E402
    CallbackBundle, KairosCheckpoint, KairosParetoGuard, KairosPendulumLR,
)
from kairos.integrations.hf import KairosHFCallback  # noqa: E402
from kairos.aletheia.pools.factuality import FactualityPool  # noqa: E402
from kairos.aletheia.pools.reasoning import ReasoningPool  # noqa: E402
from kairos.aletheia.pools.instruction import InstructionPool  # noqa: E402


MODEL_ID = "huihui-ai/Huihui-gemma-4-E2B-it-abliterated"


# ---------------------------------------------------------------------------
# Pool factory
# ---------------------------------------------------------------------------


def build_pools(tokenizer, eval_samples: int) -> list:
    """Build the 3 task-shape pools. The factuality pool's eval default
    (`basicv8/SimpleQA`) is a community reupload that may vanish; fall
    back to TriviaQA hash-split if the load fails."""
    reasoning = ReasoningPool(
        tokenizer=tokenizer,
        eval_samples=eval_samples,
        thinking_mode=False,           # base model doesn't know <think> tags
        max_new_tokens=128,
    )
    factuality = FactualityPool(
        tokenizer=tokenizer,
        train_subset="rc.nocontext",    # smaller than rc.wikipedia.nocontext
        eval_dataset_id="mandarjoshi/trivia_qa",
        eval_subset="rc.nocontext",
        eval_split="validation",        # held-out TriviaQA split
        eval_samples=eval_samples,
        max_new_tokens=32,
    )
    instruction = InstructionPool(
        tokenizer=tokenizer,
        train_dataset_id="tatsu-lab/alpaca",  # smaller than Tulu-3 for smoke
        train_subset=None,
        eval_dataset_id="tatsu-lab/alpaca",
        eval_split="train",             # last N held out via hash, see below
        eval_samples=eval_samples,
        max_new_tokens=128,
    )
    return [reasoning, factuality, instruction]


# ---------------------------------------------------------------------------
# Training data assembly
# ---------------------------------------------------------------------------


def build_mixed_dataset(pools: list, n_per_pool: int):
    """Pull n_per_pool formatted examples from each pool's
    `_format_example` (which returns prompt-target pairs), tokenize
    with proper prompt-token masking (-100 on prompt positions), and
    concatenate into a single shuffled HF Dataset.

    The pool's _format_example + _tokenize_pair give us label-masked
    labels that train only on the *target* tokens — significantly
    better signal than training on the entire concatenated string.
    """
    from datasets import Dataset

    rows: list[dict] = []
    for p in pools:
        ds = p._load_hf(p.train_dataset_id, p.train_subset, p.train_split)
        taken = 0
        for ex in ds:
            try:
                prompt, target = p._format_example(ex)
            except Exception:
                continue
            if not prompt or not target:
                continue
            tok = p._tokenize_pair(prompt, target)
            tok["pool"] = p.name
            rows.append(tok)
            taken += 1
            if taken >= n_per_pool:
                break
        print(f"  [{p.name}] kept {taken} train rows", flush=True)

    # Build pure-Python lists (HF Dataset.from_list copies)
    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=0)
    # Strip the pool tag — collator doesn't need it
    ds = ds.remove_columns(["pool"])
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--train-size-per-pool", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=16,
                          help="OOT examples per pool (generation-based)")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--floor-mult", type=float, default=0.80)
    parser.add_argument("--save-dir", type=str,
                          default="./aletheia_gemma4_v2_ckpt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run-data", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        Trainer, TrainingArguments,
    )

    print(f"loading tokenizer for {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pools = build_pools(tok, args.eval_samples)
    print(f"pools: {[p.name for p in pools]}", flush=True)

    print("\nbuilding mixed training set...", flush=True)
    train_ds = build_mixed_dataset(pools, args.train_size_per_pool)
    print(f"  mixed dataset size: {len(train_ds)}", flush=True)

    if args.dry_run_data:
        print("\n[dry-run-data] sample row tokens:")
        for i in range(min(3, len(train_ds))):
            row = train_ds[i]
            n_unmasked = sum(1 for x in row["labels"] if x != -100)
            print(f"  row {i}: input_ids len={len(row['input_ids'])} "
                  f"target_tokens={n_unmasked}")
        return 0

    print(f"\nloading {MODEL_ID} weights ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    print(f"  base loaded in {time.time()-t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # --- Baseline eval (task-shape metrics) ---
    print("\nbaseline eval (per-pool OOT, generation-based)...", flush=True)
    baseline_scores: dict[str, float] = {}
    for p in pools:
        t_eval = time.time()
        res = p.evaluate(model, batch_size=1)
        baseline_scores[p.name] = res.score
        print(f"  {p.name:>12s}: score={res.score:.4f}  "
              f"n={res.n_examples}  wall={time.time()-t_eval:.1f}s",
              flush=True)

    # --- LoRA wrap (text decoder only) ---
    import bitsandbytes.optim as bnbo
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import DataCollatorForSeq2Seq

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=(
            r".*language_model\.layers\.\d+\."
            r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
        ),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n  LoRA: trainable={n_trainable/1e6:.2f}M  "
          f"total={n_total/1e9:.3f}B  "
          f"trainable_pct={100*n_trainable/n_total:.3f}%", flush=True)

    # --- Optimiser ---
    opt = bnbo.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    # --- Kairos bundle ---
    pareto = KairosParetoGuard(
        anchor=baseline_scores, metric_prefix="pool_",
        floor_mult=args.floor_mult,
    )
    pendulum = KairosPendulumLR.for_distillation(optimizer=opt,
                                                    apply_smoothing=0.85)
    ckpt = KairosCheckpoint(save_dir=str(save_dir))
    bundle = CallbackBundle(pareto, pendulum, ckpt)
    pendulum.set_initial_lrs(args.lr)

    # --- Periodic eval (generation-based) ---
    from transformers import TrainerCallback

    class _PeriodicEval(TrainerCallback):
        def __init__(self):
            self.history: list[dict] = []
            self._step = 0

        def on_step_end(self, args_, state, control, **kwargs):
            self._step = state.global_step

        def on_log(self, args_, state, control, logs=None, **kwargs):
            logs = logs or {}
            if (self._step > 0 and self._step % args.eval_every == 0
                    and "_evaled_at" not in logs):
                model.eval()
                row: dict[str, Any] = {"step": self._step}
                axis_keys: dict[str, float] = {}
                eval_t0 = time.time()
                for p in pools:
                    res = p.evaluate(model, batch_size=1)
                    row[p.name] = res.score
                    axis_keys[f"pool_{p.name}"] = res.score
                row["eval_wall_s"] = round(time.time() - eval_t0, 1)
                row["wall_s"] = round(time.time() - t0, 1)
                self.history.append(row)
                summary = "  ".join(f"{p.name}={row[p.name]:.4f}" for p in pools)
                print(f"[eval @ step {self._step}]  {summary}  "
                      f"({row['eval_wall_s']}s)", flush=True)

                axis_keys["train_loss"] = float(logs.get("loss", 0.0))
                axis_keys["train_acc"] = 0.0
                axis_keys["test_loss"] = 0.0
                axis_keys["test_acc"] = float(
                    sum(row[p.name] for p in pools) / len(pools)
                )
                action = bundle.observe(self._step, **axis_keys)
                for note in action.notes:
                    print(f"  pareto: {note}", flush=True)
                with open(save_dir / "eval_history.jsonl", "a",
                            encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                model.train()

    periodic_eval = _PeriodicEval()

    # --- Trainer ---
    targs = TrainingArguments(
        output_dir=str(save_dir / "trainer"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=0,
        lr_scheduler_type="constant",
        logging_steps=max(1, args.eval_every // 5),
        save_steps=10_000,
        report_to=[],
        bf16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                                pad_to_multiple_of=8,
                                                return_tensors="pt",
                                                label_pad_token_id=-100),
        callbacks=[KairosHFCallback(bundle), periodic_eval],
        optimizers=(opt, None),
    )

    torch.cuda.reset_peak_memory_stats()
    print("\nstarting LoRA SFT (multi-axis Pareto, task-shape metrics)...",
          flush=True)
    trainer.train()
    print(f"done. VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    # --- Final eval ---
    print("\nfinal eval (per-pool OOT)...", flush=True)
    final_scores: dict[str, float] = {}
    for p in pools:
        res = p.evaluate(model, batch_size=1)
        final_scores[p.name] = res.score

    print()
    print("=" * 72)
    print(f"{'pool':<14}{'baseline':>12}{'final':>12}{'Δ score':>14}"
          f"{'Δ%':>10}{'verdict':>10}")
    print("-" * 72)
    for p in pools:
        b = baseline_scores[p.name]
        f = final_scores[p.name]
        d = f - b
        pct = 100 * d / b if b > 1e-6 else float("inf") if d > 0 else 0
        verdict = "↑" if d > 0 else "↓" if d < 0 else "="
        if f < args.floor_mult * b:
            verdict = "↓ FLOOR"
        pct_str = f"{pct:+.2f}%" if math.isfinite(pct) else "from zero"
        print(f"{p.name:<14}{b:>12.4f}{f:>12.4f}{d:>+14.4f}{pct_str:>10}"
              f"{verdict:>10}")
    print("=" * 72)
    prod_base = math.prod(max(v, 1e-6) for v in baseline_scores.values())
    prod_final = math.prod(max(v, 1e-6) for v in final_scores.values())
    print(f"floor_mult={args.floor_mult}, "
          f"product baseline={prod_base:.6f}, "
          f"product final={prod_final:.6f}")

    report = {
        "model_id": MODEL_ID,
        "args": vars(args),
        "baseline_scores": baseline_scores,
        "final_scores": final_scores,
        "eval_history": periodic_eval.history,
    }
    (save_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {save_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
