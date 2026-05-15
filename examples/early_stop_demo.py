"""Focused demo: KairosEarlyStop saves compute on dead-end runs.

Trains the same model twice with a configuration that NEVER groks
(weight_decay=0.0 makes the model memorise then sit forever):

  A. BASELINE_DEAD:    runs the full step budget despite no progress
  B. EARLY_STOP_DEAD:  aborts ~step 12_000 (memorisation_step
                        ~ 150 + stable_steps_to_abort = 10_000 +
                        min_step gate, etc.)

Reports wall-clock and the wall-clock SAVED by EarlyStop. This is
the clearest dollar-value demonstration of the toolkit.
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

from kairos import CallbackBundle, KairosEarlyStop
from examples.train_with_kairos import (
    GrokTransformer, build_dataset, evaluate,
)


def train_one(label: str, *, n_steps: int, weight_decay: float,
              use_early_stop: bool, device: torch.device, seed: int = 0
              ) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y, test_x, test_y = build_dataset(29, 0.3, seed, device)
    model = GrokTransformer(p=29).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                             weight_decay=weight_decay)
    bundle = None
    if use_early_stop:
        bundle = CallbackBundle(
            KairosEarlyStop(stable_steps_to_abort=10_000, min_step=2000),
        )
    history: list[dict] = []
    t0 = time.time()
    aborted_at = None
    for step in range(n_steps):
        logits = model(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        opt.zero_grad(); loss.backward(); opt.step()
        tloss = float(loss.item())
        tacc = float((logits.argmax(-1) == train_y).float().mean().item())
        ev_loss, ev_acc = evaluate(model, test_x, test_y)
        history.append({"step": step, "train_loss": tloss, "train_acc": tacc,
                         "test_loss": ev_loss, "test_acc": ev_acc})
        if bundle is not None:
            a = bundle.observe(step, train_loss=tloss, train_acc=tacc,
                                test_loss=ev_loss, test_acc=ev_acc)
            if a.stop_training:
                aborted_at = step
                print(f"[{label}] EarlyStop fired at step {step}: {a.stop_reason}")
                break
        if step % 1000 == 0:
            print(f"[{label}] step={step}  train_acc={tacc:.3f}  "
                  f"test_acc={ev_acc:.3f}  wall={time.time()-t0:.1f}s")
    wall = time.time() - t0
    return {
        "label": label,
        "n_steps_run": len(history),
        "wall_seconds": wall,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "aborted_at": aborted_at,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    n_steps = 15_000

    # Dead-end config: weight_decay=0.0 -- memorises forever, never groks.
    print("\n=== A) BASELINE_DEAD (wd=0.0, no EarlyStop) ===")
    a = train_one("A:BASELINE_DEAD", n_steps=n_steps, weight_decay=0.0,
                   use_early_stop=False, device=device)

    print("\n=== B) EARLY_STOP_DEAD (wd=0.0, with EarlyStop) ===")
    b = train_one("B:EARLY_STOP", n_steps=n_steps, weight_decay=0.0,
                   use_early_stop=True, device=device)

    print()
    print("=" * 78)
    print(f"{'label':<20} {'steps':>6} {'wall':>9} {'aborted':>8} "
          f"{'final_test_acc':>14}")
    for r in (a, b):
        print(f"{r['label']:<20} {r['n_steps_run']:>6} "
              f"{r['wall_seconds']:>8.1f}s "
              f"{str(r['aborted_at']):>8} {r['final_test_acc']:>14.3f}")
    saved = a["wall_seconds"] - b["wall_seconds"]
    pct = 100 * saved / max(a["wall_seconds"], 1e-9)
    print()
    print(f"Compute saved by EarlyStop on this dead-end run: "
          f"{saved:.1f}s ({pct:.0f}% of baseline wall clock)")

    out = pathlib.Path("early_stop_demo_results.json")
    out.write_text(json.dumps([a, b], indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
