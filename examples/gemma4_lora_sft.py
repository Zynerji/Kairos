"""LoRA SFT of huihui-ai/Huihui-gemma-4-E2B-it-abliterated on
OpenThoughts-114k R1-distill thinking traces, with Kairos in the loop
via ``KairosHFCallback`` and ``recommended_bundle("distillation")``.

This is the first real test of the Kairos -> HF Trainer integration
(mock-tested only in tests/test_integrations_mocked.py).

Design:
  * LoRA rank 8 on q/k/v/o + gate/up/down (Gemma 4 attention + MLP).
    Vision + audio encoders stay frozen — text-only SFT.
  * bitsandbytes AdamW8bit optimiser (fits 24 GB with margin).
  * HF Trainer ``lr_scheduler_type="constant"`` so KairosPendulumLR
    owns the LR signal. ``warmup_steps=0`` — no warmup needed for
    LoRA SFT on a pretrained model; this also sidesteps the
    pendulum LR-anchor trap (no warmup LR to mis-capture).
  * Refusal-filtered ``open-thoughts/OpenThoughts-114k`` rows via
    the sharegpt schema handler shared with finetune_deepseek_r1.py.

Run on the VM:
    PYTHONPATH=. python3 examples/gemma4_lora_sft.py \\
        --max-steps 50 --batch-size 1 --grad-accum 8

Outputs:
    ./gemma4_lora_ckpt/<adapter>
    ./gemma4_lora_ckpt/metrics.jsonl
    ./gemma4_lora_ckpt/generation_samples.txt
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time
from typing import Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import recommended_bundle  # noqa: E402
from kairos.integrations.hf import KairosHFCallback  # noqa: E402

# Reuse the format + refusal pipeline from the Mamba ft scaffold.
from examples.finetune_deepseek_r1 import (  # noqa: E402
    REFUSAL_REGEX, format_example,
)


MODEL_ID = "huihui-ai/Huihui-gemma-4-E2B-it-abliterated"


def build_dataset(tokenizer, seq_len: int, n_examples: int):
    """Pull `n_examples` formatted, refusal-filtered rows from
    OpenThoughts-114k and tokenise them as a flat causal-LM corpus."""
    from datasets import load_dataset, Dataset
    import torch

    print(f"streaming {n_examples} examples from open-thoughts/OpenThoughts-114k ...",
          flush=True)
    src = load_dataset("open-thoughts/OpenThoughts-114k", split="train",
                        streaming=True)

    texts: list[str] = []
    dropped_refusal = 0
    dropped_empty = 0
    for ex in src:
        text = format_example(ex)
        if text is None:
            if REFUSAL_REGEX.search(json.dumps(ex)):
                dropped_refusal += 1
            else:
                dropped_empty += 1
            continue
        texts.append(text)
        if len(texts) >= n_examples:
            break
    print(f"  kept={len(texts)}  refusal_dropped={dropped_refusal}  "
          f"empty_dropped={dropped_empty}", flush=True)

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=seq_len,
                          padding="max_length", return_tensors=None)
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50,
                          help="grad updates (NOT raw steps)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--n-examples", type=int, default=512)
    parser.add_argument("--save-dir", type=str,
                          default="./gemma4_lora_ckpt")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import bitsandbytes.optim as bnbo
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    print(f"  base loaded in {time.time()-t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB",
          flush=True)

    # Freeze the vision + audio encoders explicitly (LoRA targets text only).
    for name, p in model.named_parameters():
        if name.startswith(("model.vision_tower", "model.audio_tower",
                              "model.multi_modal_projector",
                              "model.embed_audio", "model.embed_vision")):
            p.requires_grad_(False)

    # LoRA: q/k/v/o + gate/up/down on the TEXT decoder only.
    # Subtle: Gemma 4 also has q/k/v/o in its vision + audio encoders,
    # but those are wrapped in Gemma4ClippableLinear (clamp+Linear) and
    # don't receive input during text-only forward. Targeting them with
    # plain suffix matches puts LoRA on dormant paths -> grad_norm=0,
    # no learning. So we use a regex anchored on `language_model.layers`
    # to only touch the text decoder's plain nn.Linear projections.
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
    # Required when LoRA targets inner sub-modules of frozen wrappers
    # (Gemma 4 wraps each projection in Gemma4ClippableLinear).
    # Without this, the embedding output has requires_grad=False and
    # the LoRA-wrapped inner linear's gradient input is detached.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: trainable={n_trainable/1e6:.2f}M  "
          f"total={n_total/1e9:.3f}B  "
          f"trainable_pct={100*n_trainable/n_total:.3f}%", flush=True)

    # Dataset
    ds = build_dataset(tok, args.seq_len, args.n_examples)
    print(f"  dataset size: {len(ds)}", flush=True)

    # Optimiser BEFORE bundle so pendulum can hold the reference.
    opt = bnbo.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    # Kairos bundle. pretraining/distillation profile both drive on
    # train_loss; distillation is the right framing for SFT.
    bundle = recommended_bundle("distillation", optimizer=opt,
                                  checkpoint_dir=None)
    pendulum = next((cb for cb in bundle.callbacks
                       if type(cb).__name__ == "KairosPendulumLR"), None)
    if pendulum is not None:
        # No HF warmup, so the captured-on-first-observe LR will be
        # exactly args.lr. Re-anchor anyway for safety.
        pendulum.set_initial_lrs(args.lr)

    # Per-step metrics file for offline inspection.
    metrics_path = save_dir / "metrics.jsonl"
    metrics_path.write_text("")
    metrics_log = []

    class _Recorder:
        name = "_StepRecorder"
        def __init__(self):
            self._t0 = time.time()
        def observe(self, step, monitor, **m):
            row = {
                "step": int(step),
                "train_loss": float(m.get("train_loss", float("nan"))),
                "lr": float(opt.param_groups[0]["lr"]),
                "pendulum_state": getattr(pendulum, "state", None),
                "wall_s": round(time.time() - self._t0, 2),
            }
            metrics_log.append(row)
            with open(metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            from kairos.core import Action
            return Action()
    bundle.callbacks.append(_Recorder())

    targs = TrainingArguments(
        output_dir=str(save_dir / "trainer"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=0,
        lr_scheduler_type="constant",
        logging_steps=args.log_every,
        save_steps=10_000,             # don't bother saving during smoke
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
        callbacks=[KairosHFCallback(bundle)],
        optimizers=(opt, None),
    )

    torch.cuda.reset_peak_memory_stats()
    print("starting LoRA SFT ...", flush=True)
    trainer.train()
    print(f"done. VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)

    # Save LoRA adapter
    adapter_dir = save_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    print(f"saved adapter -> {adapter_dir}", flush=True)

    # Generation samples (LoRA active)
    print("\n=== generation samples (LoRA active) ===", flush=True)
    model.eval()
    prompts = [
        "<start_of_turn>user\nWhat is 7 + 5? Think step by step.<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWrite a Python function to reverse a string.<end_of_turn>\n<start_of_turn>model\n",
    ]
    out_path = save_dir / "generation_samples.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=80, do_sample=False,
                                       repetition_penalty=1.1)
            text = tok.decode(out[0], skip_special_tokens=False)
            tail = text[len(p):]
            print(f"PROMPT: {p}\nOUTPUT: {tail}\n", flush=True)
            fh.write(f"PROMPT: {p}\nOUTPUT: {tail}\n---\n")
    print(f"wrote samples -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
