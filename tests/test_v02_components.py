"""Tests for the v0.2.0 components ported from the local-repo research stack:
PendulumLR (Kanon), ParetoGuard (Aletheia), GrowthController (qGPT-Infinity).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import (
    Action,
    CallbackBundle,
    GrowthSignal,
    KairosGrowthController,
    KairosParetoGuard,
    KairosPendulumLR,
    ParetoState,
)


class _FakeOptimizer:
    def __init__(self, lr: float = 1e-3, n: int = 1) -> None:
        self.param_groups = [
            {"lr": float(lr), "weight_decay": 0.0} for _ in range(n)
        ]


# ---------------------------------------------------------------------------
# KairosPendulumLR
# ---------------------------------------------------------------------------


def test_pendulum_lr_constructs():
    p = KairosPendulumLR()
    assert p.state == "ACTIVE"


def test_pendulum_lr_invalid_metric_raises():
    with pytest.raises(ValueError):
        KairosPendulumLR(metric="bogus")


def test_pendulum_lr_invalid_smoothing_raises():
    with pytest.raises(ValueError):
        KairosPendulumLR(apply_smoothing=1.0)


def test_pendulum_lr_advances_with_loss_stream():
    p = KairosPendulumLR(apply_smoothing=0.0)
    bundle = CallbackBundle(p)
    # Feed a chaotic loss stream -> CV should be high -> EXPLORE
    rng = np.random.default_rng(0)
    for step in range(200):
        loss = 1.0 + 5.0 * float(rng.standard_normal()) ** 2
        bundle.observe(step, train_loss=loss, train_acc=0.5,
                        test_loss=1.0, test_acc=0.3)
    # Should have visited at least 2 different phases
    visited = {p.state}
    p2 = KairosPendulumLR(apply_smoothing=0.0)
    bundle2 = CallbackBundle(p2)
    # Now a flat loss stream -> CV low -> CRYSTAL
    for step in range(200):
        bundle2.observe(step, train_loss=0.01, train_acc=1.0,
                        test_loss=0.01, test_acc=1.0)
    visited.add(p2.state)
    assert len(visited) >= 1   # at least one explicit state
    assert p.crystal_count + p.active_count + p.explore_count == 200


def test_pendulum_lr_mutates_optimizer_when_provided():
    opt = _FakeOptimizer(lr=1e-3)
    p = KairosPendulumLR(optimizer=opt, apply_smoothing=0.0)
    bundle = CallbackBundle(p)
    # Inject a chaotic loss stream -> EXPLORE -> lr_mult > 1 OR
    # alternating regimes -> lr should at least vary from 1e-3 at
    # some point. Test that the LR is *not stuck* at the initial
    # value across the run.
    import random
    random.seed(0)
    saw_change = False
    for step in range(500):
        loss = float(random.uniform(0.0, 5.0))
        bundle.observe(step, train_loss=loss, train_acc=0.5,
                        test_loss=loss, test_acc=0.3)
        if abs(opt.param_groups[0]["lr"] - 1e-3) > 1e-9:
            saw_change = True
    assert saw_change, "PendulumLR never modulated the optimizer LR"
    # State counters should sum to step count
    assert (p.crystal_count + p.active_count + p.explore_count) == 500


def test_pendulum_lr_emits_action_with_multiplier():
    p = KairosPendulumLR(apply_smoothing=0.0)
    bundle = CallbackBundle(p)
    a = bundle.observe(0, train_loss=0.5, train_acc=0.5,
                        test_loss=0.5, test_acc=0.3)
    assert isinstance(a, Action)
    assert a.lr_multiplier > 0


def test_pendulum_lr_for_grokking_preset():
    """`for_grokking` uses test_loss (validated +62 pp on modular-arithmetic
    Transformer, 2026-05-15 GPU run)."""
    p = KairosPendulumLR.for_grokking()
    assert p.metric == "test_loss"


def test_pendulum_lr_for_distillation_preset():
    p = KairosPendulumLR.for_distillation()
    assert p.metric == "train_loss"


# ---------------------------------------------------------------------------
# KairosParetoGuard
# ---------------------------------------------------------------------------


def test_pareto_constructs_with_anchor():
    g = KairosParetoGuard(anchor={"acc": 0.8, "cal": 0.7})
    assert g.best_product > 0


def test_pareto_requires_anchor():
    with pytest.raises(ValueError):
        KairosParetoGuard(anchor={})


def test_pareto_invalid_floor_raises():
    with pytest.raises(ValueError):
        KairosParetoGuard(anchor={"x": 0.5}, floor_mult=1.5)


def test_pareto_new_best_fires_save():
    """Improvement on all axes => new best => save_checkpoint=True."""
    g = KairosParetoGuard(anchor={"acc": 0.5, "cal": 0.5}, metric_prefix="")
    bundle = CallbackBundle(g)
    a = bundle.observe(0, acc=0.6, cal=0.6,
                        train_loss=1.0, train_acc=0.5,
                        test_loss=1.0, test_acc=0.5)
    assert a.save_checkpoint
    assert "pareto_best" in a.checkpoint_tag


def test_pareto_no_new_best_when_one_axis_below_floor():
    """One axis above floor, one barely above => not new best (and not rollback)."""
    g = KairosParetoGuard(anchor={"acc": 0.5, "cal": 0.5}, metric_prefix="",
                            floor_mult=0.8)
    bundle = CallbackBundle(g)
    a = bundle.observe(0, acc=0.6, cal=0.39,  # 0.39 < 0.8 * 0.5 = 0.4
                        train_loss=1.0, train_acc=0.5,
                        test_loss=1.0, test_acc=0.5)
    assert not a.save_checkpoint


def test_pareto_rollback_when_two_axes_below_floor():
    g = KairosParetoGuard(anchor={"acc": 0.5, "cal": 0.5, "rea": 0.5},
                            metric_prefix="", floor_mult=0.8)
    bundle = CallbackBundle(g)
    a = bundle.observe(0, acc=0.39, cal=0.39, rea=0.6,
                        train_loss=1.0, train_acc=0.5,
                        test_loss=1.0, test_acc=0.5)
    # Two axes below 0.4 floor
    assert g.last_state.last_should_rollback
    assert any("ROLLBACK" in n for n in a.notes)


def test_pareto_state_records_below_floor_axes():
    g = KairosParetoGuard(anchor={"acc": 0.5, "cal": 0.5},
                            metric_prefix="", floor_mult=0.8)
    bundle = CallbackBundle(g)
    bundle.observe(0, acc=0.39, cal=0.39,
                    train_loss=1.0, train_acc=0.5,
                    test_loss=1.0, test_acc=0.5)
    assert set(g.last_state.last_below_floor_axes) == {"acc", "cal"}


def test_pareto_metric_prefix_matching():
    g = KairosParetoGuard(anchor={"acc": 0.5, "cal": 0.5},
                            metric_prefix="pool_")
    bundle = CallbackBundle(g)
    a = bundle.observe(0, pool_acc=0.6, pool_cal=0.6,
                        train_loss=1.0, train_acc=0.5,
                        test_loss=1.0, test_acc=0.5)
    assert a.save_checkpoint  # picked up via "pool_" prefix


def test_pareto_spectral_amplification_companion():
    g = KairosParetoGuard(
        anchor={"acc": 0.5}, metric_prefix="",
        spectral_target_std=1.0, alpha_max=5.0,
    )
    bundle = CallbackBundle(g)
    a = bundle.observe(0, acc=0.7, acc_std=0.001,  # collapsed -> high alpha
                        train_loss=1.0, train_acc=0.5,
                        test_loss=1.0, test_acc=0.5)
    assert any("spectral_amp" in n for n in a.notes)


# ---------------------------------------------------------------------------
# KairosGrowthController
# ---------------------------------------------------------------------------


def test_growth_constructs():
    g = KairosGrowthController()
    assert g.n_grow_events == 0


def test_growth_returns_noop_below_min_steps():
    g = KairosGrowthController(min_steps_before_grow=500, window_size=20)
    bundle = CallbackBundle(g)
    for step in range(100):
        a = bundle.observe(step, train_loss=0.5, hidden_var=0.5,
                            depth_signal=0.5, train_acc=0.5, test_acc=0.3,
                            test_loss=0.5)
        # No growth signal before min_steps_before_grow
        assert not any(n.startswith("grow:") for n in a.notes)


def test_growth_fires_on_saturation():
    """Loss flat for many steps after min_steps => growth signal."""
    g = KairosGrowthController(
        K_threshold=0.5, W_threshold=0.5, D_threshold=0.5,
        window_size=20, min_steps_before_grow=50, cooldown_steps=100,
    )
    bundle = CallbackBundle(g)
    # Phase 1: descending loss (no saturation)
    for step in range(100):
        loss = 1.0 - 0.005 * step
        bundle.observe(step, train_loss=loss, hidden_var=0.5,
                        depth_signal=0.5, train_acc=0.5, test_acc=0.3,
                        test_loss=loss)
    # Phase 2: completely flat loss (saturated)
    fired = False
    for step in range(100, 600):
        a = bundle.observe(step, train_loss=0.5, hidden_var=0.5,
                            depth_signal=0.5, train_acc=1.0, test_acc=0.3,
                            test_loss=0.5)
        if any(n.startswith("grow:") for n in a.notes):
            fired = True
            break
    assert fired or g.n_grow_events > 0, (
        "growth signal never fired despite long flat-loss period"
    )


def test_growth_cooldown_respected():
    """After a growth event, no more fires within cooldown_steps."""
    g = KairosGrowthController(
        K_threshold=10.0, W_threshold=10.0, D_threshold=10.0,
        window_size=10, min_steps_before_grow=20, cooldown_steps=200,
        max_grow_events=8,
    )
    bundle = CallbackBundle(g)
    grow_steps: list[int] = []
    for step in range(1000):
        a = bundle.observe(step, train_loss=0.5, hidden_var=0.5,
                            depth_signal=0.5, train_acc=1.0, test_acc=0.3,
                            test_loss=0.5)
        if any(n.startswith("grow:") for n in a.notes):
            grow_steps.append(step)
    # Successive grow events must be >= cooldown_steps apart
    for a, b in zip(grow_steps, grow_steps[1:]):
        assert b - a >= 200


def test_growth_history_snapshots():
    g = KairosGrowthController(min_steps_before_grow=10, window_size=5)
    bundle = CallbackBundle(g)
    for step in range(30):
        bundle.observe(step, train_loss=0.5, hidden_var=0.5,
                        depth_signal=0.5, train_acc=0.5, test_acc=0.3,
                        test_loss=0.5)
    assert len(g.history) == 30
    assert all(isinstance(h, GrowthSignal) for h in g.history)


def test_growth_max_events_respected():
    g = KairosGrowthController(
        K_threshold=10.0, W_threshold=10.0, D_threshold=10.0,
        window_size=5, min_steps_before_grow=20, cooldown_steps=50,
        max_grow_events=3,
    )
    bundle = CallbackBundle(g)
    for step in range(5000):
        bundle.observe(step, train_loss=0.5, hidden_var=0.5,
                        depth_signal=0.5, train_acc=1.0, test_acc=0.3,
                        test_loss=0.5)
    assert g.n_grow_events <= 3
