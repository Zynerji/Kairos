"""KairosParetoGuard demo on a simulated multi-axis training run.

Synthesises a 3-axis training trajectory (factuality / calibration /
reasoning, all in [0, 1]) and streams it through ParetoGuard:

  - axis 1 (factuality):  rises steadily from 0.6 -> 0.9
  - axis 2 (calibration): rises but dips around step 400-500
  - axis 3 (reasoning):   rises slowly with bigger dip mid-run
  - around step 600: simultaneous dip in 2+3 -> ROLLBACK signal

Reports:
  - the steps at which new-best checkpoints fired
  - the steps at which rollback signals fired
  - the final product metric vs the anchor
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import CallbackBundle, KairosParetoGuard


def main():
    rng = np.random.default_rng(0)
    n_steps = 1000

    anchor = {"factuality": 0.60, "calibration": 0.55, "reasoning": 0.50}

    def trajectory(step: int) -> dict[str, float]:
        """Generate (axis -> score) per step."""
        # Factuality: monotone rising
        fact = 0.60 + 0.30 * step / n_steps + 0.01 * rng.standard_normal()
        # Calibration: rising with a single mid-run dip
        cal_base = 0.55 + 0.25 * step / n_steps
        cal_dip = -0.20 if 400 < step < 500 else 0.0
        cal = cal_base + cal_dip + 0.01 * rng.standard_normal()
        # Reasoning: rising with a larger overlapping dip
        rea_base = 0.50 + 0.30 * step / n_steps
        rea_dip = -0.25 if 550 < step < 700 else 0.0
        rea = rea_base + rea_dip + 0.01 * rng.standard_normal()
        return {"factuality": fact, "calibration": cal, "reasoning": rea}

    guard = KairosParetoGuard(anchor=anchor, floor_mult=0.80, metric_prefix="")
    bundle = CallbackBundle(guard)
    print(f"anchor: {anchor}")
    print(f"floor_mult: 0.80 (rollback when >= 2 axes < 0.80 * anchor)")
    print()
    print(f"{'step':>6} {'fact':>5} {'cal':>5} {'rea':>5} {'product':>9}  event")
    events: list[dict] = []
    for step in range(n_steps):
        scores = trajectory(step)
        a = bundle.observe(
            step,
            train_loss=0.5, train_acc=0.5, test_loss=0.5, test_acc=0.5,
            **scores,
        )
        if a.save_checkpoint or any("ROLLBACK" in n for n in a.notes):
            event = ("SAVE_BEST" if a.save_checkpoint
                     else "ROLLBACK")
            product = guard.last_state.last_product
            print(f"{step:>6} {scores['factuality']:>5.3f} "
                  f"{scores['calibration']:>5.3f} {scores['reasoning']:>5.3f} "
                  f"{product:>9.4f}  {event}")
            events.append({"step": step, "event": event,
                            "scores": scores,
                            "below_floor": list(guard.last_state.last_below_floor_axes),
                            "product": product})

    print()
    print("=" * 78)
    print(f"  n_new_best signals (checkpoints fired): {guard.n_new_best_signals}")
    print(f"  n_rollback signals: {guard.n_rollback_signals}")
    print(f"  best_product: {guard.best_product:.4f}")
    print(f"  best_scores : {guard.best_scores}")

    out = pathlib.Path("pareto_guard_demo_results.json")
    out.write_text(json.dumps({
        "anchor": anchor, "events": events,
        "best_product": guard.best_product,
        "best_scores": guard.best_scores,
        "n_new_best": guard.n_new_best_signals,
        "n_rollback": guard.n_rollback_signals,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
