"""End-to-end training run with the full Kairos stack engaged.

Trains a 2-layer Transformer on modular arithmetic, with all seven
Kairos components wired into the training loop:

  1. KairosEarlyStop      -- aborts if memorisation-only for N steps
  2. KairosLRSchedule     -- drops LR at the grokking transition
  3. KairosCheckpoint     -- saves model at the transition
  4. (KairosSweepGate)    -- demo'd separately; needs multi-trial harness
  5. KairosAccelerator    -- perturbation pulses during the plateau
  6. KairosCurriculum     -- phase-aware lr/wd schedule
  7. (KairosProbe)        -- demo'd separately; multi-capability scores

Compares two configurations:
  * BASELINE: no callbacks; raw training
  * KAIROS  : all 5 in-loop callbacks active

Reports for each:
  - whether grokking happened and at what step
  - wall-clock seconds
  - final test_acc
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import (
    CallbackBundle,
    KairosAccelerator,
    KairosCheckpoint,
    KairosCurriculum,
    KairosEarlyStop,
    KairosLRSchedule,
)


# ---------------------------------------------------------------------------
# Modular-arithmetic data
# ---------------------------------------------------------------------------


def build_dataset(p: int, train_frac: float, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    perm = rng.permutation(len(pairs))
    n_train = int(len(pairs) * train_frac)
    ti, te = perm[:n_train], perm[n_train:]
    return (
        torch.from_numpy(pairs[ti]).to(device),
        torch.from_numpy(labels[ti]).to(device),
        torch.from_numpy(pairs[te]).to(device),
        torch.from_numpy(labels[te]).to(device),
    )


class GrokTransformer(nn.Module):
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, ff: int = 512) -> None:
        super().__init__()
        self.p = p
        self.emb = nn.Embedding(p + 1, d_model)
        self.pos = nn.Embedding(3, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            batch_first=True, activation="gelu",
        )
        self.body = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, p)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        B = pairs.shape[0]
        sep = torch.full((B, 1), self.p, dtype=torch.long, device=pairs.device)
        x = torch.cat([pairs, sep], dim=1)
        pos = torch.arange(3, device=x.device).unsqueeze(0).expand(B, 3)
        h = self.emb(x) + self.pos(pos)
        h = self.body(h)
        return self.head(h[:, -1, :])


# ---------------------------------------------------------------------------
# Train one configuration
# ---------------------------------------------------------------------------


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    model.train()
    return float(loss), float(acc)


def train_once(
    *,
    label: str,
    p: int,
    train_frac: float,
    n_steps: int,
    seed: int,
    use_kairos: bool,
    device: torch.device,
    checkpoint_dir: pathlib.Path | None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y, test_x, test_y = build_dataset(p, train_frac, seed, device)

    model = GrokTransformer(p=p).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"[{label}] params={n_params:,}  device={device}  p={p}  "
          f"train={len(train_x)}  test={len(test_x)}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    bundle = None
    if use_kairos == "full":
        bundle = CallbackBundle(
            KairosEarlyStop(stable_steps_to_abort=20_000, min_step=2000),
            KairosLRSchedule(drop_factor=0.1, optimizer=opt),
            KairosCheckpoint(save_dir=checkpoint_dir),
            KairosAccelerator(sigma=0.005, wait_steps=2000,
                              cooldown_steps=2000, max_pulses=8, auto_apply=True),
            KairosCurriculum(optimizer=opt),
        )
    elif use_kairos == "conservative":
        # The shippable subset: EarlyStop + LRSchedule + Checkpoint.
        # No accelerator (perturbation), no curriculum (phase-flapping
        # was hurting on the slow-grok regime).
        bundle = CallbackBundle(
            KairosEarlyStop(stable_steps_to_abort=20_000, min_step=2000),
            KairosLRSchedule(drop_factor=0.1, optimizer=opt),
            KairosCheckpoint(save_dir=checkpoint_dir),
        )

    history: list[dict] = []
    grok_step = None
    t0 = time.time()
    log_every = 200
    last_phase_logged = None
    for step in range(n_steps):
        logits = model(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss = float(loss.item())
        train_acc = float((logits.argmax(-1) == train_y).float().mean().item())
        test_loss, test_acc = evaluate(model, test_x, test_y)
        history.append({"step": step, "train_loss": train_loss,
                         "train_acc": train_acc,
                         "test_loss": test_loss, "test_acc": test_acc})

        if bundle is not None:
            action = bundle.observe(
                step, train_loss=train_loss, train_acc=train_acc,
                test_loss=test_loss, test_acc=test_acc, model=model,
            )
            if action.notes and step != last_phase_logged:
                for n in action.notes:
                    print(f"[{label}] step={step}: {n}")
                last_phase_logged = step
            if grok_step is None and bundle.monitor.detected_event is not None:
                grok_step = step
                print(f"[{label}] *** GROK detected at step {step} ***")
            if action.stop_training:
                print(f"[{label}] EARLY STOP at step {step}: {action.stop_reason}")
                break

        if step % log_every == 0 or step == n_steps - 1:
            wall = time.time() - t0
            phase = bundle.current_phase.value if bundle else "?"
            print(f"[{label}] step={step:>5}  train_loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  test_loss={test_loss:.4f}  "
                  f"test_acc={test_acc:.3f}  wall={wall:.1f}s  phase={phase}")
    wall = time.time() - t0

    final = history[-1]
    return {
        "label": label,
        "n_steps_run": len(history),
        "wall_seconds": wall,
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "grok_step": grok_step,
        "history_tail": history[-200:],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=29)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--n-steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="kairos_checkpoints")
    parser.add_argument("--mode",
                        choices=("all", "baseline", "kairos_full", "kairos_conservative"),
                        default="all")
    parser.add_argument("--out", type=str,
                        default="train_with_kairos_results.json")
    args = parser.parse_args()

    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if args.mode in ("baseline", "all"):
        results.append(train_once(
            label="BASELINE", p=args.p, train_frac=args.train_frac,
            n_steps=args.n_steps, seed=args.seed, use_kairos=None,
            device=device, checkpoint_dir=None,
        ))
    if args.mode in ("kairos_conservative", "all"):
        results.append(train_once(
            label="KAIROS-C", p=args.p, train_frac=args.train_frac,
            n_steps=args.n_steps, seed=args.seed, use_kairos="conservative",
            device=device, checkpoint_dir=ckpt_dir / "conservative",
        ))
    if args.mode in ("kairos_full", "all"):
        results.append(train_once(
            label="KAIROS-F", p=args.p, train_frac=args.train_frac,
            n_steps=args.n_steps, seed=args.seed, use_kairos="full",
            device=device, checkpoint_dir=ckpt_dir / "full",
        ))

    print()
    print("=" * 78)
    print(f"{'label':<10} {'steps':>6} {'wall':>9} {'grok@':>8} "
          f"{'final_test_acc':>14}")
    for r in results:
        print(f"{r['label']:<10} {r['n_steps_run']:>6} "
              f"{r['wall_seconds']:>8.1f}s "
              f"{str(r['grok_step']):>8} {r['final_test_acc']:>14.3f}")

    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
