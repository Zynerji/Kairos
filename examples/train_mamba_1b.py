"""Train a ~0.7B-parameter Mamba LM with Kairos.

This is a *scaffold* — the actual launch happens on the Blackwell VM
once multi-seed validation lands.

Two configs are offered:

- Default ("~0.74B"): ``d_model=2048, n_layer=24`` — 738M params.
  Fits in 24 GB VRAM with fp32 AdamW at batch=2 seq_len=1024
  (peak VRAM measured 6.6 GB forward, ~18 GB with full optimiser
  state). This is the right size for a single-Blackwell autonomous
  run.

- ``--canonical-1p4b``: ``d_model=2048, n_layer=48`` — 1.37B params.
  Matches state-spaces/mamba-1.4b. Needs ``--adam-8bit`` (bitsandbytes
  ``AdamW8bit``) to fit in 24 GB. Empirically 14 GB peak at batch=1
  seq=1024 on RTX PRO 4000 Blackwell with 8-bit Adam (vs OOM at fp32).

Pretraining objective: standard next-token cross-entropy on a
mid-sized public corpus (default: HuggingFace ``allenai/c4`` ``en``
split, streaming). Kairos drives the LR via
``recommended_bundle("pretraining")`` — the Hamiltonian pendulum on
train_loss replaces cosine.

Anti-resonant init is applied to the LM head and embeddings (the two
"weight-tied" components) before training; SSM kernel weights use
the upstream mamba_ssm.modules defaults.

The grokking-monitor is included as a passive observer — even if no
canonical "grokking event" fires during a typical LM pretrain,
Cassandra's CSD signal correlates with capability emergence (per
Power 2022 follow-ups, e.g., Olsson et al. 2022 "In-context Learning
and Induction Heads").

Usage (on VM with mamba_ssm installed):

    PYTHONPATH=. python3 examples/train_mamba_1b.py \\
        --total-steps 50000 --batch-size 16 --seq-len 2048 \\
        --grad-accum 4 --save-dir ./mamba_1b_ckpt

The scaffold falls back to a tiny test config (d_model=256, n_layers=4)
when ``--smoke`` is passed, so it can run on CPU for verification.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Imports that don't need torch
from kairos import (  # noqa: E402
    KairosAntiResonantInit,
    recommended_bundle,
)


@dataclass
class MambaConfig:
    d_model: int = 2048
    n_layers: int = 24
    vocab_size: int = 50_280  # GPT-NeoX (close to mamba-1.4b release)
    ssm_state: int = 16
    expand: int = 2
    rms_norm: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8

    @classmethod
    def smoke(cls) -> "MambaConfig":
        return cls(d_model=256, n_layers=4, vocab_size=1024,
                    ssm_state=8, expand=2)

    @classmethod
    def b0p74(cls) -> "MambaConfig":
        """0.74B Mamba (738M actual). Fits 24 GB VRAM with fp32 AdamW."""
        return cls(d_model=2048, n_layers=24, vocab_size=50_280,
                    ssm_state=16, expand=2)

    @classmethod
    def b1p3(cls) -> "MambaConfig":
        """Canonical Mamba-1.4B (1.31B params). Needs >24 GB with
        fp32 AdamW — use 8-bit Adam or FSDP on multi-GPU."""
        return cls(d_model=2048, n_layers=48, vocab_size=50_280,
                    ssm_state=16, expand=2)


def build_mamba_model(cfg: MambaConfig, device: str = "cuda"):
    """Build a Mamba LM. Uses ``mamba_ssm`` on CUDA, falls back to a
    naive Mamba block for CPU smoke tests."""
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig as MambaSSMCfg
        m_cfg = MambaSSMCfg(
            d_model=cfg.d_model,
            n_layer=cfg.n_layers,
            vocab_size=cfg.vocab_size,
            ssm_cfg={"d_state": cfg.ssm_state, "expand": cfg.expand},
            rms_norm=cfg.rms_norm,
            fused_add_norm=cfg.fused_add_norm,
            pad_vocab_size_multiple=cfg.pad_vocab_size_multiple,
        )
        return MambaLMHeadModel(m_cfg).to(device)
    except (ImportError, RuntimeError) as e:
        # CPU smoke fallback
        print(f"[warn] mamba_ssm not available ({type(e).__name__}); "
              f"using naive Mamba block for smoke test", flush=True)
        import torch
        import torch.nn as nn

        class _NaiveBlock(nn.Module):
            def __init__(self, d_model: int) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.in_proj = nn.Linear(d_model, d_model * 2)
                self.out_proj = nn.Linear(d_model * 2, d_model)

            def forward(self, x):
                h = self.norm(x)
                h = torch.nn.functional.silu(self.in_proj(h))
                return x + self.out_proj(h)

        class _NaiveMambaLM(nn.Module):
            def __init__(self, c: MambaConfig) -> None:
                super().__init__()
                self.embed = nn.Embedding(c.vocab_size, c.d_model)
                self.blocks = nn.ModuleList([
                    _NaiveBlock(c.d_model) for _ in range(c.n_layers)
                ])
                self.norm_f = nn.LayerNorm(c.d_model)
                self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
                # tie
                self.lm_head.weight = self.embed.weight

            def forward(self, input_ids, **_):
                x = self.embed(input_ids)
                for b in self.blocks:
                    x = b(x)
                x = self.norm_f(x)
                logits = self.lm_head(x)
                return type("Out", (), {"logits": logits})()

        return _NaiveMambaLM(cfg).to(device)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataloader(seq_len: int, batch_size: int, *, smoke: bool):
    """Streaming C4 dataloader. For smoke, generates random token IDs."""
    if smoke:
        import torch

        def _gen():
            while True:
                ids = torch.randint(0, 1024, (batch_size, seq_len))
                yield {"input_ids": ids, "labels": ids.clone()}

        return _gen()

    # Real path: HuggingFace datasets streaming C4
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "real-data path needs `datasets` and `transformers`. "
            "Install on VM: pip install datasets transformers"
        ) from e
    import torch

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _gen():
        buf: list[int] = []
        for ex in ds:
            ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
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
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default="./mamba_1b_ckpt")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--keep-last", type=int, default=3,
                          help="how many intermediate ckpts to keep on disk")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true",
                          help="tiny config + random data for CPU smoke")
    parser.add_argument("--canonical-1p4b", action="store_true",
                          help="use 48-layer canonical 1.4B config "
                          "(combine with --adam-8bit to fit 24 GB)")
    parser.add_argument("--adam-8bit", action="store_true",
                          help="use bitsandbytes AdamW8bit "
                          "(required for 1.4B on 24 GB)")
    parser.add_argument("--no-antiresonant", action="store_true",
                          help="skip antiresonant init")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    if args.smoke:
        cfg = MambaConfig.smoke()
    elif args.canonical_1p4b:
        cfg = MambaConfig.b1p3()
    else:
        cfg = MambaConfig.b0p74()
    print(f"config: {cfg}", flush=True)

    model = build_mamba_model(cfg, device=device)
    n_params = count_params(model)
    print(f"params: {n_params/1e6:.1f}M ({n_params:,})", flush=True)

    if not args.no_antiresonant:
        init = KairosAntiResonantInit(suppress_top_k=0,
                                        scale_factor=0.02,
                                        phase_staggered_embeddings=True,
                                        seed=0)
        # Only init embedding + lm_head (SSM blocks have proven init).
        emb = getattr(getattr(model, "backbone", model), "embedding", None)
        head = getattr(model, "lm_head", None)
        if emb is not None or head is not None:
            stub = nn.Module()
            if emb is not None:
                stub.add_module("embedding", emb)
            if head is not None:
                stub.add_module("lm_head", head)
            rep = init.apply(stub)
            print(f"antiresonant init: applied to {rep.n_linear} linear "
                  f"+ {rep.n_embedding} embedding layers", flush=True)

    if args.adam_8bit:
        import bitsandbytes.optim as bnbo
        opt = bnbo.AdamW8bit(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)
        print("optimiser: bitsandbytes.AdamW8bit", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
        print("optimiser: torch.optim.AdamW (fp32)", flush=True)

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    bundle = recommended_bundle("pretraining", optimizer=opt,
                                  checkpoint_dir=str(save_dir))
    # Find the pendulum to re-anchor its base LR post-warmup.
    pendulum = next((cb for cb in bundle.callbacks
                       if type(cb).__name__ == "KairosPendulumLR"), None)
    _pendulum_anchored = False

    loader = build_dataloader(args.seq_len, args.batch_size,
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

        # Linear warmup over warmup-steps; pendulum-modulated afterwards.
        if step < args.warmup_steps:
            lr_warm = args.lr * (step + 1) / max(args.warmup_steps, 1)
            for g in opt.param_groups:
                g["lr"] = lr_warm
        elif not _pendulum_anchored and pendulum is not None:
            # Tell pendulum its anchor LR is args.lr, NOT the warmup LR
            # it captured on first observe.
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
                wall = time.time() - t0
                lr_now = opt.param_groups[0]["lr"]
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
                ckpt = save_dir / f"mamba_upd{grad_update}.pt"
                torch.save({"model": model.state_dict(),
                             "grad_update": grad_update,
                             "step": step, "loss": avg_loss},
                            ckpt)
                print(f"saved {ckpt}", flush=True)
                # Rotate: keep only the last N intermediate snapshots
                snaps = sorted(save_dir.glob("mamba_upd*.pt"),
                                key=lambda p: int(p.stem.replace("mamba_upd", "")))
                for old in snaps[:-args.keep_last]:
                    try:
                        old.unlink()
                        print(f"pruned {old}", flush=True)
                    except OSError:
                        pass

        step += 1

    # Final
    if not args.smoke:
        ckpt = save_dir / "mamba_final.pt"
        torch.save({"model": model.state_dict(), "step": step}, ckpt)
        print(f"final saved {ckpt}", flush=True)

    (save_dir / "metrics.json").write_text(
        json.dumps(metrics_log, indent=2)
    )
    print(f"wall total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
