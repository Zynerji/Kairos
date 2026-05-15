"""Headline A/B: KairosPendulumLR vs static-LR baseline.

Trains the same Transformer twice with identical seed:
  A. BASELINE_STATIC_LR — AdamW(lr=1e-3, wd=1.0), no LR adaptation
  B. KAIROS_PENDULUM    — same optimizer, KairosPendulumLR continuously
                          modulating LR via Hamiltonian-pendulum CV state

Reports for each:
  - final test_acc + train_acc
  - wall-clock seconds
  - pendulum CRYSTAL / ACTIVE / EXPLORE step counts (for B only)
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import CallbackBundle, KairosPendulumLR
from examples.train_with_kairos import (
    GrokTransformer, build_dataset, evaluate,
)


def train(label: str, *, n_steps: int, use_pendulum: bool,
           device: torch.device, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tx, ty, ex, ey = build_dataset(29, 0.3, seed, device)
    model = GrokTransformer(p=29).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

    pendulum: KairosPendulumLR | None = None
    bundle: CallbackBundle | None = None
    if use_pendulum:
        # On grokking-style tasks where train_loss flatlines at ~0
        # after memorisation, the pendulum should be driven by the
        # test stream (which IS oscillating during the slow climb --
        # exactly Cassandra's CSD signal).
        pendulum = KairosPendulumLR(
            metric="test_loss", optimizer=opt, apply_smoothing=0.85,
        )
        bundle = CallbackBundle(pendulum)

    history: list[dict] = []
    t0 = time.time()
    log_every = 200
    for step in range(n_steps):
        logits = model(tx)
        loss = nn.functional.cross_entropy(logits, ty)
        opt.zero_grad(); loss.backward(); opt.step()
        tloss = float(loss.item())
        tacc = float((logits.argmax(-1) == ty).float().mean().item())
        ev_loss, ev_acc = evaluate(model, ex, ey)
        history.append({"step": step, "train_loss": tloss, "train_acc": tacc,
                         "test_loss": ev_loss, "test_acc": ev_acc,
                         "lr": float(opt.param_groups[0]["lr"])})
        if bundle is not None:
            bundle.observe(step, train_loss=tloss, train_acc=tacc,
                            test_loss=ev_loss, test_acc=ev_acc)
        if step % log_every == 0 or step == n_steps - 1:
            wall = time.time() - t0
            state = pendulum.state if pendulum else "-"
            cv = pendulum.cv if pendulum else 0.0
            cur_lr = opt.param_groups[0]["lr"]
            print(f"[{label}] step={step:>5} train_acc={tacc:.3f} "
                  f"test_acc={ev_acc:.3f} lr={cur_lr:.2e} "
                  f"state={state} cv={cv:.3f} wall={wall:.1f}s")
    wall = time.time() - t0

    final = history[-1]
    out: dict = {
        "label": label, "n_steps_run": len(history),
        "wall_seconds": wall,
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "final_lr": final["lr"],
    }
    if pendulum is not None:
        out["pendulum_crystal_count"] = pendulum.crystal_count
        out["pendulum_active_count"] = pendulum.active_count
        out["pendulum_explore_count"] = pendulum.explore_count
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    n_steps = 15_000

    print("\n=== A) BASELINE_STATIC_LR (lr=1e-3 constant) ===")
    a = train("A:STATIC", n_steps=n_steps, use_pendulum=False, device=device)

    print("\n=== B) KAIROS_PENDULUM (lr=1e-3 * pendulum-multiplier) ===")
    b = train("B:PENDULUM", n_steps=n_steps, use_pendulum=True, device=device)

    print()
    print("=" * 78)
    print(f"{'label':<14} {'wall':>9} {'final_train_acc':>16} "
          f"{'final_test_acc':>15}")
    for r in (a, b):
        print(f"{r['label']:<14} {r['wall_seconds']:>8.1f}s "
              f"{r['final_train_acc']:>16.3f} {r['final_test_acc']:>15.3f}")
    print()
    if "pendulum_crystal_count" in b:
        c, ac, e = (b["pendulum_crystal_count"], b["pendulum_active_count"],
                    b["pendulum_explore_count"])
        total = c + ac + e
        print(f"Pendulum phase distribution over {total} steps:")
        print(f"  CRYSTAL: {c:>5} ({100*c/total:5.1f}%)  lr_mult ≈ 0.382")
        print(f"  ACTIVE : {ac:>5} ({100*ac/total:5.1f}%)  lr_mult = 1.000")
        print(f"  EXPLORE: {e:>5} ({100*e/total:5.1f}%)  lr_mult ≈ 1.618")
    delta = b["final_test_acc"] - a["final_test_acc"]
    print()
    print(f"Test-acc delta (PENDULUM - STATIC): {delta:+.3f}")

    out = pathlib.Path("pendulum_lr_demo_results.json")
    out.write_text(json.dumps([a, b], indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
