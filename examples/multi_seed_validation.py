"""Multi-seed validation of PendulumLR.for_grokking().

Runs 5 seeds x 2 configs (baseline static-LR vs KairosPendulumLR
driven by test_loss). Reports mean +/- std of final test accuracy
and per-seed table.

If the +62 pp delta from the single-seed run is robust, this should
show pendulum >> baseline at every seed (or a clear majority). If
it's bimodal, that tells us the pendulum is catalysing grokking
when seed is favourable. If it's just noise, the means will be
within std of each other.
"""

from __future__ import annotations

import json
import pathlib
import statistics
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


def train_once(seed: int, *, n_steps: int, use_pendulum: bool,
               device: torch.device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tx, ty, ex, ey = build_dataset(29, 0.3, seed, device)
    model = GrokTransformer(p=29).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    bundle = None
    if use_pendulum:
        bundle = CallbackBundle(
            KairosPendulumLR.for_grokking(optimizer=opt),
        )
    t0 = time.time()
    history: list[dict] = []
    for step in range(n_steps):
        logits = model(tx)
        loss = nn.functional.cross_entropy(logits, ty)
        opt.zero_grad(); loss.backward(); opt.step()
        tl = float(loss.item())
        ta = float((logits.argmax(-1) == ty).float().mean().item())
        evl, eva = evaluate(model, ex, ey)
        history.append({"step": step, "train_loss": tl, "train_acc": ta,
                         "test_loss": evl, "test_acc": eva,
                         "lr": float(opt.param_groups[0]["lr"])})
        if bundle is not None:
            bundle.observe(step, train_loss=tl, train_acc=ta,
                            test_loss=evl, test_acc=eva)
    wall = time.time() - t0
    return {"seed": seed, "use_pendulum": use_pendulum,
             "wall_seconds": wall,
             "final_train_acc": history[-1]["train_acc"],
             "final_test_acc": history[-1]["test_acc"],
             "best_test_acc_anywhere": max(h["test_acc"] for h in history)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    seeds = [0, 1, 2, 3, 4]
    n_steps = 15_000

    rows: list[dict] = []
    for seed in seeds:
        for use_p in (False, True):
            label = "PENDULUM" if use_p else "BASELINE"
            print(f"\n[seed={seed} {label}] starting...")
            r = train_once(seed, n_steps=n_steps, use_pendulum=use_p,
                            device=device)
            print(f"[seed={seed} {label}] wall={r['wall_seconds']:.1f}s  "
                  f"final_test_acc={r['final_test_acc']:.3f}  "
                  f"best_test_acc={r['best_test_acc_anywhere']:.3f}")
            rows.append(r)

    # Aggregate
    base = [r for r in rows if not r["use_pendulum"]]
    pend = [r for r in rows if r["use_pendulum"]]
    base_finals = [r["final_test_acc"] for r in base]
    pend_finals = [r["final_test_acc"] for r in pend]
    base_bests = [r["best_test_acc_anywhere"] for r in base]
    pend_bests = [r["best_test_acc_anywhere"] for r in pend]

    print()
    print("=" * 78)
    print(f"{'config':<14} {'mean_final':>11} {'std_final':>10} "
          f"{'mean_best':>10} {'std_best':>9}  {'per_seed_final'}")
    for label, finals, bests in [
        ("BASELINE", base_finals, base_bests),
        ("PENDULUM", pend_finals, pend_bests),
    ]:
        m_f, s_f = statistics.mean(finals), statistics.stdev(finals)
        m_b, s_b = statistics.mean(bests), statistics.stdev(bests)
        per_seed = "  ".join(f"{x:.3f}" for x in finals)
        print(f"{label:<14} {m_f:>11.3f} {s_f:>10.3f} "
              f"{m_b:>10.3f} {s_b:>9.3f}  {per_seed}")

    delta_finals = [p - b for p, b in zip(pend_finals, base_finals)]
    delta_mean = statistics.mean(delta_finals)
    delta_std = statistics.stdev(delta_finals)
    print()
    print(f"Paired delta (PENDULUM - BASELINE):")
    print(f"  per seed: {[round(x, 3) for x in delta_finals]}")
    print(f"  mean +- std: {delta_mean:+.3f} +- {delta_std:.3f}")

    pendulum_wins = sum(1 for d in delta_finals if d > 0.05)
    print(f"  pendulum >> baseline at {pendulum_wins} / {len(seeds)} seeds (delta > +0.05)")

    out = pathlib.Path("multi_seed_validation_results.json")
    out.write_text(json.dumps({
        "n_seeds": len(seeds), "n_steps": n_steps, "rows": rows,
        "summary": {
            "baseline_mean_final": statistics.mean(base_finals),
            "baseline_std_final": statistics.stdev(base_finals),
            "pendulum_mean_final": statistics.mean(pend_finals),
            "pendulum_std_final": statistics.stdev(pend_finals),
            "paired_delta_mean": delta_mean,
            "paired_delta_std": delta_std,
            "pendulum_wins": pendulum_wins,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
