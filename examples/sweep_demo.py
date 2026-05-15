"""KairosSweepGate demo: 4 hparam trials, kill the laggards at step 2000.

Trains 4 modular-arithmetic Transformers in lockstep with different
weight-decay values (the load-bearing grokking hparam). At step 2000
KairosSweepGate scores each and keeps the top-2; the other 2 are
abandoned. Reports the final outcome of the survivors.
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

from kairos import KairosSweepGate
from examples.train_with_kairos import (  # noqa: E402 -- sibling import
    GrokTransformer,
    build_dataset,
    evaluate,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = 29
    train_frac = 0.3
    n_steps = 10_000
    eval_at_step = 2000
    keep_top_k = 2

    hparams = [
        {"name": f"wd_{wd:.2f}", "wd": wd, "lr": 1e-3}
        for wd in (0.0, 0.1, 0.5, 1.0)
    ]
    print(f"sweep: {len(hparams)} trials, keep top {keep_top_k} at step "
          f"{eval_at_step}, train for {n_steps}")

    # Initialise each trial
    trials = {}
    for hp in hparams:
        torch.manual_seed(0)
        np.random.seed(0)
        tx, ty, ex, ey = build_dataset(p, train_frac, seed=0, device=device)
        model = GrokTransformer(p=p).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        trials[hp["name"]] = {"model": model, "opt": opt,
                              "tx": tx, "ty": ty, "ex": ex, "ey": ey,
                              "hp": hp, "history": [], "alive": True,
                              "grok_step": None}

    gate = KairosSweepGate(n_trials=len(hparams), eval_at_step=eval_at_step,
                            keep_top_k=keep_top_k)
    t0 = time.time()
    log_every = 200
    for step in range(n_steps):
        for name, tr in trials.items():
            if not tr["alive"]:
                continue
            logits = tr["model"](tr["tx"])
            loss = nn.functional.cross_entropy(logits, tr["ty"])
            tr["opt"].zero_grad(); loss.backward(); tr["opt"].step()
            train_loss = float(loss.item())
            train_acc = float((logits.argmax(-1) == tr["ty"]).float().mean().item())
            test_loss, test_acc = evaluate(tr["model"], tr["ex"], tr["ey"])
            tr["history"].append({"step": step, "train_loss": train_loss,
                                    "train_acc": train_acc,
                                    "test_loss": test_loss,
                                    "test_acc": test_acc})
            gate.observe_trial(name, step, train_loss=train_loss,
                                train_acc=train_acc, test_loss=test_loss,
                                test_acc=test_acc)
        if step == eval_at_step:
            decisions = gate.make_decisions()
            print(f"\n=== Gate decision at step {step} ===")
            for name, d in sorted(decisions.items(),
                                   key=lambda x: -x[1].score):
                tag = "KILL" if d.kill else "KEEP"
                print(f"  {tag}  {name:<10}  score={d.score:.3f}  "
                      f"reason={d.reason}")
                if d.kill:
                    trials[name]["alive"] = False
            print()
        if step % log_every == 0 or step == n_steps - 1:
            wall = time.time() - t0
            alive_acc = {n: tr["history"][-1]["test_acc"]
                          for n, tr in trials.items() if tr["alive"]}
            print(f"step={step:>5} wall={wall:>6.1f}s alive_test_acc="
                  f"{ {k: round(v, 3) for k, v in alive_acc.items()} }")

    print()
    print("=" * 78)
    print(f"{'trial':<10} {'alive':>6} {'final_acc':>10} {'wd':>6}")
    for name, tr in trials.items():
        final = tr["history"][-1]["test_acc"] if tr["history"] else float("nan")
        print(f"{name:<10} {'yes' if tr['alive'] else 'no':>6} "
              f"{final:>10.3f} {tr['hp']['wd']:>6}")

    out = pathlib.Path("sweep_demo_results.json")
    out.write_text(json.dumps({
        "n_steps": n_steps, "eval_at_step": eval_at_step,
        "keep_top_k": keep_top_k,
        "results": {name: {"hp": tr["hp"], "alive": tr["alive"],
                            "final_test_acc": (tr["history"][-1]["test_acc"]
                                                if tr["history"] else None),
                            "n_history": len(tr["history"])}
                     for name, tr in trials.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
