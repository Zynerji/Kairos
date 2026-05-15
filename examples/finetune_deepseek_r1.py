"""Fine-tune the pretrained 1.3B Mamba on DeepSeek-R1-Distill thinking
traces.

DeepSeek-R1-Distill datasets are MIT/Apache-2 reasoning traces
emitted by R1 (or R1-Distill-Qwen) on math/code/logic problems. The
canonical public corpora:

  - open-thoughts/OpenThoughts-114k
  - bespokelabs/Bespoke-Stratos-17k
  - simplescaling/s1K-1.1
  - open-r1/OpenR1-Math-220k

This script loads a Mamba checkpoint from
``examples/train_mamba_1b.py`` and runs supervised fine-tuning on
``<thinking_traces> <answer>`` pairs. Kairos drives the LR via
``recommended_bundle("distillation")`` — fine-tuning has the
continuously-falling train_loss shape that ``for_distillation()``
was tuned for in Aletheia.

Refusal filtering (per Aletheia + memory feedback "Refusal-filter
ruthlessly before ingestion"): regex-based gate drops examples whose
thinking traces contain refusal language.

Usage (VM):

    PYTHONPATH=. python3 examples/finetune_deepseek_r1.py \\
        --base-ckpt mamba_1b_ckpt/mamba_final.pt \\
        --dataset open-thoughts/OpenThoughts-114k \\
        --total-steps 10000 --save-dir ./mamba_ft_ckpt

Smoke (CPU):

    python examples/finetune_deepseek_r1.py --smoke --total-steps 4
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import recommended_bundle  # noqa: E402

# Conservative refusal patterns — drop trace if any match.
REFUSAL_REGEX = re.compile(
    r"(?i)\b("
    r"I can't|I cannot|I'm sorry, but|I am sorry, but|As an AI|"
    r"I'm not able to|I am not able to|I shouldn't|I should not|"
    r"It would be inappropriate|It would be unsafe|"
    r"I am unable to|I'm unable to|"
    r"I am not capable|I'm not capable"
    r")\b"
)


def filter_refusals(text: str) -> bool:
    """Return True if text passes the refusal filter (i.e., NOT a refusal)."""
    return not bool(REFUSAL_REGEX.search(text or ""))


def format_example(ex: dict) -> str | None:
    """Map a HF row to a single ``prompt + thinking + answer`` string.
    Returns None if the row should be dropped (refusal / empty)."""
    # Try common DeepSeek-R1-Distill schemas in priority order.
    if "messages" in ex:
        parts = []
        for m in ex["messages"]:
            role, content = m.get("role"), m.get("content", "")
            if not content:
                continue
            parts.append(f"<|{role}|>\n{content}")
        text = "\n".join(parts)
    elif "conversations" in ex:
        # sharegpt-style (OpenThoughts, Bespoke-Stratos, s1K, ...)
        parts = []
        sys = ex.get("system")
        if isinstance(sys, str) and sys:
            parts.append(f"<|system|>\n{sys}")
        for m in ex["conversations"]:
            role = m.get("from") or m.get("role")
            val = m.get("value") or m.get("content", "")
            if not val or not role:
                continue
            # normalise user/human → user, gpt/assistant → assistant
            role = {"human": "user", "gpt": "assistant"}.get(role, role)
            parts.append(f"<|{role}|>\n{val}")
        text = "\n".join(parts)
    elif "question" in ex and "answer" in ex:
        thinking = ex.get("reasoning") or ex.get("thinking") or ""
        text = (f"<|user|>\n{ex['question']}\n<|assistant|>\n"
                f"<thinking>{thinking}</thinking>\n{ex['answer']}")
    elif "problem" in ex and "solution" in ex:
        text = (f"<|user|>\n{ex['problem']}\n<|assistant|>\n"
                f"{ex['solution']}")
    elif "instruction" in ex:
        out = ex.get("output") or ex.get("response", "")
        text = f"<|user|>\n{ex['instruction']}\n<|assistant|>\n{out}"
    else:
        # Fall back: concatenate string fields
        text = "\n".join(str(v) for v in ex.values()
                          if isinstance(v, str) and v)
        if not text:
            return None

    if not filter_refusals(text):
        return None
    return text


def build_loader(dataset_name: str, seq_len: int, batch_size: int,
                  *, smoke: bool):
    if smoke:
        import torch

        def _gen():
            while True:
                ids = torch.randint(0, 1024, (batch_size, seq_len))
                yield {"input_ids": ids, "labels": ids.clone()}

        return _gen()

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "real-data path needs `datasets` and `transformers`."
        ) from e
    import torch

    ds = load_dataset(dataset_name, split="train", streaming=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _gen():
        buf: list[int] = []
        n_dropped_refusal = 0
        n_dropped_empty = 0
        for ex in ds:
            text = format_example(ex)
            if text is None:
                if REFUSAL_REGEX.search(json.dumps(ex)):
                    n_dropped_refusal += 1
                else:
                    n_dropped_empty += 1
                continue
            ids = tok(text, add_special_tokens=False,
                       truncation=True, max_length=seq_len)["input_ids"]
            buf.extend(ids + [tok.eos_token_id])
            while len(buf) >= batch_size * seq_len:
                chunk = buf[:batch_size * seq_len]
                buf = buf[batch_size * seq_len:]
                t = torch.tensor(chunk, dtype=torch.long).view(batch_size,
                                                                 seq_len)
                yield {"input_ids": t, "labels": t.clone()}

    return _gen()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, default=None,
                          help="path to pretrained Mamba checkpoint")
    parser.add_argument("--dataset", type=str,
                          default="open-thoughts/OpenThoughts-114k")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="./mamba_ft_ckpt")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-last", type=int, default=2,
                          help="how many intermediate ckpts to keep on disk")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    from train_mamba_1b import MambaConfig, build_mamba_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    cfg = MambaConfig.smoke() if args.smoke else MambaConfig.b0p74()
    model = build_mamba_model(cfg, device=device)

    if args.base_ckpt is not None and not args.smoke:
        state = torch.load(args.base_ckpt, map_location=device)
        if "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded {args.base_ckpt}: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fine-tuning: train_loss is the live signal (Aletheia-proven).
    bundle = recommended_bundle("distillation", optimizer=opt,
                                  checkpoint_dir=str(save_dir))
    pendulum = next((cb for cb in bundle.callbacks
                       if type(cb).__name__ == "KairosPendulumLR"), None)
    _pendulum_anchored = False

    loader = build_loader(args.dataset, args.seq_len, args.batch_size,
                            smoke=args.smoke)

    model.train()
    step = 0
    grad_update = 0
    t0 = time.time()
    metrics_log: list[dict] = []
    grad_accum_loss = 0.0

    for batch in loader:
        if step >= args.total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if step < args.warmup_steps:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / max(args.warmup_steps, 1)
        elif not _pendulum_anchored and pendulum is not None:
            pendulum.set_initial_lrs(args.lr)
            _pendulum_anchored = True

        out = model(input_ids)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        (loss / args.grad_accum).backward()
        grad_accum_loss += float(loss.item())

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            grad_update += 1
            avg_loss = grad_accum_loss / args.grad_accum
            grad_accum_loss = 0.0
            ppl = math.exp(min(avg_loss, 20))
            bundle.observe(step, train_loss=avg_loss, train_acc=0.0,
                            test_loss=avg_loss, test_acc=0.0)
            if grad_update % max(args.log_every, 1) == 0:
                lr_now = opt.param_groups[0]["lr"]
                wall = time.time() - t0
                print(f"upd={grad_update:>6} step={step:>6}  "
                      f"loss={avg_loss:.4f}  ppl={ppl:>8.2f}  "
                      f"lr={lr_now:.2e}  wall={wall:.0f}s",
                      flush=True)
                metrics_log.append({
                    "grad_update": grad_update, "step": step,
                    "loss": avg_loss, "ppl": ppl,
                    "lr": lr_now, "wall_seconds": wall,
                })
            if grad_update > 0 and grad_update % args.save_every == 0 and not args.smoke:
                ckpt = save_dir / f"mamba_ft_upd{grad_update}.pt"
                torch.save({"model": model.state_dict(),
                             "grad_update": grad_update,
                             "step": step, "loss": avg_loss}, ckpt)
                print(f"saved {ckpt}", flush=True)
                # Rotate: keep only the last N intermediate snapshots
                snaps = sorted(save_dir.glob("mamba_ft_upd*.pt"),
                                key=lambda p: int(p.stem.replace("mamba_ft_upd", "")))
                for old in snaps[:-args.keep_last]:
                    try:
                        old.unlink()
                        print(f"pruned {old}", flush=True)
                    except OSError:
                        pass

        step += 1

    if not args.smoke:
        torch.save({"model": model.state_dict(), "step": step},
                    save_dir / "mamba_ft_final.pt")
        print(f"final saved {save_dir / 'mamba_ft_final.pt'}", flush=True)

    (save_dir / "metrics.json").write_text(
        json.dumps(metrics_log, indent=2)
    )
    print(f"wall total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
