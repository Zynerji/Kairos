"""Aletheia-style multi-axis Pareto LoRA heal for an abliterated LLM.

What this is
============
A minimal but honest port of the Aletheia post-training protocol
(see Aletheia/CLAUDE.md), scoped to a single LoRA adapter on the
text decoder of a multimodal HF model. It does NOT yet implement
per-pool LoRA + torsion cycling — that's Phase B and a separate
scaffold. This is Phase A: prove the multi-axis Pareto loop drives
SFT without regressing any axis below the 80% anchor floor.

Mechanism
=========
1. **Baseline eval** of the base abliterated model on held-out slices
   of N pools. Each pool's metric = ``exp(-mean_loss)`` on the held-
   out slice (higher = better, in (0, 1]).
2. **Anchor**: pool baseline scores set the ParetoGuard anchors.
3. **Mixed-batch SFT**: each training step samples uniformly from one
   pool. LoRA adapters wrap only the text decoder (per Gemma 4
   ``Gemma4ClippableLinear`` lesson from gemma4_lora_sft.py).
4. **Periodic per-pool OOT eval** every ``--eval-every`` grad updates.
   Each eval pass updates ``KairosParetoGuard`` with the latest per-
   axis scores. The guard maintains the Pareto frontier (product
   metric on the axes), saves a checkpoint on new bests, and emits a
   ROLLBACK note when 2+ axes drop below the 80% floor.
5. **Final eval + per-axis comparison table** vs baseline.

Datasets (all public, refusal-filtered)
=======================================
- reasoning:   open-thoughts/OpenThoughts-114k (R1-distill thinking traces)
- factuality:  trivia_qa (rc.nocontext)
- instruction: tatsu-lab/alpaca

Three pools is enough to demonstrate the Pareto pattern. Adding
calibration / abstention / etc. is a copy-and-extend.

Usage on the VM
===============
    PYTHONPATH=. python3 examples/aletheia_gemma4_heal.py \\
        --max-steps 100 --batch-size 1 --grad-accum 8 \\
        --eval-every 25 --eval-size 32 --train-size-per-pool 256

Smoke locally (the model build will OOM on CPU; use --dry-run-data to
just validate the dataset loaders + pool mixing logic).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import (  # noqa: E402
    Action, CallbackBundle, KairosCheckpoint, KairosParetoGuard,
    KairosPendulumLR,
)
from kairos.integrations.hf import KairosHFCallback  # noqa: E402

from examples.finetune_deepseek_r1 import (  # noqa: E402
    REFUSAL_REGEX, format_example,
)


MODEL_ID = "huihui-ai/Huihui-gemma-4-E2B-it-abliterated"


# ---------------------------------------------------------------------------
# Pool definitions
# ---------------------------------------------------------------------------


@dataclass
class Pool:
    name: str
    hf_id: str
    hf_config: str | None = None
    split: str = "train"
    formatter: Any = None         # row dict -> string | None
    train_idx: list[int] = field(default_factory=list)
    eval_idx: list[int] = field(default_factory=list)


def fmt_openthoughts(ex: dict) -> str | None:
    return format_example(ex)  # already handles sharegpt conversations


def fmt_trivia_qa(ex: dict) -> str | None:
    q = ex.get("question")
    a_obj = ex.get("answer") or {}
    a = a_obj.get("value") if isinstance(a_obj, dict) else None
    if not q or not a:
        return None
    text = (f"<start_of_turn>user\nAnswer concisely: {q}<end_of_turn>\n"
            f"<start_of_turn>model\n{a}<end_of_turn>")
    if REFUSAL_REGEX.search(text):
        return None
    return text


def fmt_alpaca(ex: dict) -> str | None:
    instr = ex.get("instruction")
    inp = ex.get("input") or ""
    out = ex.get("output")
    if not instr or not out:
        return None
    user = instr + (f"\n\n{inp}" if inp else "")
    text = (f"<start_of_turn>user\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n{out}<end_of_turn>")
    if REFUSAL_REGEX.search(text):
        return None
    return text


POOLS: list[Pool] = [
    Pool(name="reasoning",   hf_id="open-thoughts/OpenThoughts-114k",
         formatter=fmt_openthoughts),
    Pool(name="factuality",  hf_id="trivia_qa", hf_config="rc.nocontext",
         formatter=fmt_trivia_qa),
    Pool(name="instruction", hf_id="tatsu-lab/alpaca",
         formatter=fmt_alpaca),
]


# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------


def collect_pool(pool: Pool, n_train: int, n_eval: int, seed: int) -> tuple[list[str], list[str]]:
    """Stream `n_train + n_eval` formatted, refusal-filtered rows from
    a pool. First `n_train` go to train, next `n_eval` to held-out."""
    from datasets import load_dataset

    print(f"[{pool.name}] streaming {n_train + n_eval} rows from "
          f"{pool.hf_id}...", flush=True)
    kwargs = dict(split=pool.split, streaming=True)
    if pool.hf_config is not None:
        kwargs["name"] = pool.hf_config
    ds = load_dataset(pool.hf_id, **kwargs)

    rng = random.Random(seed)
    rows: list[str] = []
    dropped_refusal = 0
    dropped_empty = 0
    # We don't shuffle a streaming dataset; just take the prefix.
    for ex in ds:
        text = pool.formatter(ex)
        if text is None:
            if REFUSAL_REGEX.search(json.dumps(ex)):
                dropped_refusal += 1
            else:
                dropped_empty += 1
            continue
        rows.append(text)
        if len(rows) >= n_train + n_eval:
            break
    rng.shuffle(rows)
    print(f"  kept={len(rows)}  refusal_dropped={dropped_refusal}  "
          f"empty_dropped={dropped_empty}", flush=True)
    return rows[:n_train], rows[n_train:n_train + n_eval]


# ---------------------------------------------------------------------------
# Per-pool eval
# ---------------------------------------------------------------------------


def pool_score(model, tokenizer, eval_texts: list[str], seq_len: int,
                device: str) -> tuple[float, float]:
    """Held-out next-token cross-entropy on the pool's eval slice.

    Returns ``(score, mean_loss)`` where score = ``exp(-mean_loss)`` so
    higher = better and the value is in (0, 1]. ParetoGuard wants
    "higher better" axes.
    """
    import torch
    model.eval()
    total_loss = 0.0
    total_count = 0
    for text in eval_texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=seq_len).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(ids)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="sum",
        )
        n_tok = shift_labels.numel()
        total_loss += float(loss.item())
        total_count += n_tok
    mean_loss = total_loss / max(total_count, 1)
    score = math.exp(-min(mean_loss, 20))   # clamp for stability
    return score, mean_loss


# ---------------------------------------------------------------------------
# Trainer integration — multi-pool dataset
# ---------------------------------------------------------------------------


def build_mixed_dataset(pools: list[Pool], tokenizer, seq_len: int):
    """Build a single HF Dataset by concatenating each pool's train
    set and tagging rows with the pool name. The Trainer samples
    uniformly across pools because we shuffle the concatenation."""
    from datasets import Dataset

    all_texts: list[str] = []
    all_pools: list[str] = []
    train_rows_by_pool: dict[str, list[str]] = {p.name: [] for p in pools}
    for p, (train, _eval) in zip(pools, pools_data):  # noqa: F821
        all_texts.extend(train)
        all_pools.extend([p.name] * len(train))
        train_rows_by_pool[p.name] = train

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=seq_len,
                          padding="max_length", return_tensors=None)
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    ds = Dataset.from_dict({"text": all_texts, "pool": all_pools})
    ds = ds.shuffle(seed=0)
    ds = ds.map(tokenize, batched=True, remove_columns=["text", "pool"])
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
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=25,
                          help="grad updates between OOT evals")
    parser.add_argument("--floor-mult", type=float, default=0.80,
                          help="Pareto floor: per-axis score must stay "
                          ">= floor_mult * baseline_anchor")
    parser.add_argument("--save-dir", type=str,
                          default="./aletheia_gemma4_ckpt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run-data", action="store_true",
                          help="just verify pool loaders, skip model + train")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Pool data ---
    global pools_data
    pools_data = []
    for p in POOLS:
        train, ev = collect_pool(p, args.train_size_per_pool,
                                    args.eval_size, args.seed)
        pools_data.append((train, ev))
        p.train_idx = list(range(len(train)))
        p.eval_idx = list(range(len(ev)))

    if args.dry_run_data:
        print("\n[dry-run-data] sample rows per pool:")
        for p, (train, ev) in zip(POOLS, pools_data):
            print(f"  {p.name}: train={len(train)} eval={len(ev)}")
            print(f"    head: {train[0][:120]!r}")
        return 0

    import bitsandbytes.optim as bnbo
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # --- Model ---
    print(f"\nloading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    print(f"  base loaded in {time.time()-t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # --- Baseline eval (anchors) ---
    print("\nbaseline eval (per-pool OOT)...", flush=True)
    baseline_scores: dict[str, float] = {}
    baseline_losses: dict[str, float] = {}
    for p, (_, ev) in zip(POOLS, pools_data):
        score, mean_loss = pool_score(model, tok, ev, args.seq_len, "cuda")
        baseline_scores[p.name] = score
        baseline_losses[p.name] = mean_loss
        print(f"  {p.name:>12s}: score={score:.4f}  loss={mean_loss:.4f}",
              flush=True)

    # --- LoRA wrap (text decoder only — see gemma4_lora_sft.py for why) ---
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
    print(f"  LoRA: trainable={n_trainable/1e6:.2f}M  "
          f"total={n_total/1e9:.3f}B  "
          f"trainable_pct={100*n_trainable/n_total:.3f}%", flush=True)

    # --- Training data ---
    ds = build_mixed_dataset(POOLS, tok, args.seq_len)
    print(f"  mixed dataset size: {len(ds)}", flush=True)

    # --- Optimiser ---
    opt = bnbo.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    # --- Kairos bundle: ParetoGuard with baseline anchors + Pendulum ---
    pareto = KairosParetoGuard(
        anchor=baseline_scores,
        metric_prefix="pool_",
        floor_mult=args.floor_mult,
    )
    pendulum = KairosPendulumLR.for_distillation(optimizer=opt,
                                                    apply_smoothing=0.85)
    ckpt = KairosCheckpoint(save_dir=str(save_dir))
    bundle = CallbackBundle(pareto, pendulum, ckpt)
    pendulum.set_initial_lrs(args.lr)

    # Periodic OOT eval bound to KairosHFCallback via a custom TrainerCallback
    from transformers import TrainerCallback

    class _PeriodicEval(TrainerCallback):
        def __init__(self):
            self.history: list[dict] = []
            self._step = 0

        def on_step_end(self, args_, state, control, **kwargs):
            self._step = state.global_step

        def on_log(self, args_, state, control, logs=None, **kwargs):
            logs = logs or {}
            # Every eval_every grad updates, run per-pool eval and feed
            # to the bundle (pareto + pendulum + checkpoint).
            if (self._step > 0 and self._step % args.eval_every == 0
                    and "_evaled_at" not in logs):
                model.eval()
                row: dict[str, float] = {}
                axis_keys: dict[str, float] = {}
                for p, (_, ev) in zip(POOLS, pools_data):
                    s, l = pool_score(model, tok, ev, args.seq_len, "cuda")
                    row[p.name] = s
                    row[f"{p.name}_loss"] = l
                    axis_keys[f"pool_{p.name}"] = s
                row["step"] = self._step
                row["wall_s"] = round(time.time() - t0, 2)
                self.history.append(row)
                # Print compact summary
                summary = "  ".join(f"{k}={v:.4f}" for k, v in row.items()
                                       if k not in ("step", "wall_s")
                                       and not k.endswith("_loss"))
                print(f"[eval @ step {self._step}]  {summary}", flush=True)
                # Feed to Kairos bundle (with the latest train_loss)
                axis_keys["train_loss"] = float(logs.get("loss", 0.0))
                axis_keys["train_acc"] = 0.0
                axis_keys["test_loss"] = float(
                    sum(row[f"{p.name}_loss"] for p in POOLS) / len(POOLS)
                )
                axis_keys["test_acc"] = float(
                    sum(row[p.name] for p in POOLS) / len(POOLS)
                )
                action = bundle.observe(self._step, **axis_keys)
                for note in action.notes:
                    print(f"  pareto: {note}", flush=True)
                # Persist row
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
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        callbacks=[KairosHFCallback(bundle), periodic_eval],
        optimizers=(opt, None),
    )

    torch.cuda.reset_peak_memory_stats()
    print("\nstarting LoRA SFT (multi-axis Pareto)...", flush=True)
    trainer.train()
    print(f"done. VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    # --- Final eval ---
    print("\nfinal eval (per-pool OOT)...", flush=True)
    final_scores: dict[str, float] = {}
    final_losses: dict[str, float] = {}
    for p, (_, ev) in zip(POOLS, pools_data):
        score, mean_loss = pool_score(model, tok, ev, args.seq_len, "cuda")
        final_scores[p.name] = score
        final_losses[p.name] = mean_loss

    # --- Comparison table ---
    print()
    print("=" * 68)
    print(f"{'pool':<14}{'baseline':>12}{'final':>12}{'Δ score':>14}"
          f"{'Δ%':>10}{'verdict':>10}")
    print("-" * 68)
    for p in POOLS:
        b = baseline_scores[p.name]
        f = final_scores[p.name]
        d = f - b
        pct = 100 * d / b if b > 0 else 0
        verdict = "↑" if d > 0 else "↓" if d < 0 else "="
        if f < args.floor_mult * b:
            verdict = "↓ FLOOR"
        print(f"{p.name:<14}{b:>12.4f}{f:>12.4f}{d:>+14.4f}{pct:>+9.2f}%"
              f"{verdict:>10}")
    print("=" * 68)
    print(f"floor_mult={args.floor_mult}, "
          f"product baseline={math.prod(baseline_scores.values()):.6f}, "
          f"product final={math.prod(final_scores.values()):.6f}")

    # --- Persist final report ---
    report = {
        "model_id": MODEL_ID,
        "args": vars(args),
        "baseline_scores": baseline_scores,
        "baseline_losses": baseline_losses,
        "final_scores": final_scores,
        "final_losses": final_losses,
        "eval_history": periodic_eval.history,
    }
    (save_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {save_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
