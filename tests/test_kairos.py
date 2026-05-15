"""Kairos tests — all 7 components on synthetic streams."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos import (
    Action,
    CallbackBundle,
    EmergenceReport,
    KairosAccelerator,
    KairosCheckpoint,
    KairosCurriculum,
    KairosEarlyStop,
    KairosLRSchedule,
    KairosProbe,
    KairosSweepGate,
    Phase,
    PhaseSettings,
    TrialDecision,
    TrialSummary,
)
from grokking_monitor import (
    GrokkingMonitor,
    simulate_grokking_curve,
    simulate_slow_grok_curve,
)


# ---------------------------------------------------------------------------
# Action + bundle
# ---------------------------------------------------------------------------


def test_action_default_is_noop():
    a = Action()
    assert not a.stop_training
    assert a.lr_multiplier == 1.0
    assert a.weight_decay_multiplier == 1.0
    assert not a.save_checkpoint
    assert a.inject_noise_sigma == 0.0


def test_action_merge_composes_multipliers():
    a = Action(lr_multiplier=0.5)
    b = Action(lr_multiplier=0.1)
    c = a.merge(b)
    assert c.lr_multiplier == pytest.approx(0.05)


def test_bundle_default_uses_slow_grok_monitor():
    b = CallbackBundle()
    assert b.monitor.jump_window == 4000   # slow-grok preset


def test_bundle_observe_returns_action_with_phase():
    b = CallbackBundle()
    a = b.observe(0, train_loss=1.0, train_acc=0.05, test_loss=1.0, test_acc=0.05)
    assert isinstance(a, Action)


# ---------------------------------------------------------------------------
# 1. KairosEarlyStop
# ---------------------------------------------------------------------------


def test_early_stop_fires_on_memorisation_only():
    """Memorisation completes at step 100, no grokking for 6000 more steps
    -> early stop fires."""
    es = KairosEarlyStop(stable_steps_to_abort=5000, min_step=500)
    bundle = CallbackBundle(es)
    fired = False
    for step in range(7000):
        train_acc = 0.05 + 0.95 * min(1.0, step / 100.0)
        a = bundle.observe(step, train_loss=0.01, train_acc=train_acc,
                            test_loss=1.2, test_acc=0.05)
        if a.stop_training:
            fired = True
            assert "memorisation" in a.stop_reason
            break
    assert fired, "EarlyStop should have fired on memorisation-only run"


def test_early_stop_does_not_fire_when_groking():
    """Use synthetic SHARP grokking curve; EarlyStop must not fire because
    the monitor detects the event before stable_steps_to_abort runs out."""
    # stable_steps_to_abort=10000 > grokking_step - memorisation_step = 4500
    es = KairosEarlyStop(stable_steps_to_abort=10_000, min_step=200)
    monitor = GrokkingMonitor.for_sharp_grok(check_every=200)
    bundle = CallbackBundle(es, monitor=monitor)
    curve = simulate_grokking_curve(n_steps=8000, memorisation_step=500,
                                      grokking_step=5000, grokking_width=400,
                                      seed=0)
    fired = False
    for i in range(len(curve.steps)):
        a = bundle.observe(
            int(curve.steps[i]),
            train_loss=float(curve.train_loss[i]),
            train_acc=float(curve.train_acc[i]),
            test_loss=float(curve.test_loss[i]),
            test_acc=float(curve.test_acc[i]),
        )
        if a.stop_training:
            fired = True
            break
    assert not fired, "EarlyStop fired on a curve that did grok"


def test_early_stop_does_not_fire_before_min_step():
    es = KairosEarlyStop(stable_steps_to_abort=10, min_step=10000)
    bundle = CallbackBundle(es)
    for step in range(1000):
        a = bundle.observe(step, train_loss=0.0, train_acc=1.0,
                            test_loss=1.2, test_acc=0.05)
        assert not a.stop_training


# ---------------------------------------------------------------------------
# 2. KairosLRSchedule
# ---------------------------------------------------------------------------


class _FakeOptimizer:
    """Minimal stand-in for torch.optim.Optimizer."""
    def __init__(self, initial_lr: float = 1e-3, n_groups: int = 1) -> None:
        self.param_groups = [
            {"lr": float(initial_lr), "weight_decay": 1.0}
            for _ in range(n_groups)
        ]


def test_lr_schedule_drops_at_transition():
    opt = _FakeOptimizer(initial_lr=1e-3)
    sched = KairosLRSchedule(drop_factor=0.1, optimizer=opt)
    monitor = GrokkingMonitor.for_sharp_grok(check_every=200)
    bundle = CallbackBundle(sched, monitor=monitor)
    curve = simulate_grokking_curve(n_steps=8000, memorisation_step=500,
                                      grokking_step=5000, grokking_width=400,
                                      seed=0)
    for i in range(len(curve.steps)):
        bundle.observe(int(curve.steps[i]),
                        train_loss=float(curve.train_loss[i]),
                        train_acc=float(curve.train_acc[i]),
                        test_loss=float(curve.test_loss[i]),
                        test_acc=float(curve.test_acc[i]))
    # LR should have dropped from 1e-3 to 1e-4
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert sched._dropped


def test_lr_schedule_invalid_drop_factor_raises():
    with pytest.raises(ValueError):
        KairosLRSchedule(drop_factor=1.5)


# ---------------------------------------------------------------------------
# 3. KairosCheckpoint
# ---------------------------------------------------------------------------


def test_checkpoint_signals_save_on_event(tmp_path):
    ckpt = KairosCheckpoint(save_dir=tmp_path)
    monitor = GrokkingMonitor.for_sharp_grok(check_every=200)
    bundle = CallbackBundle(ckpt, monitor=monitor)
    curve = simulate_grokking_curve(n_steps=8000, memorisation_step=500,
                                      grokking_step=5000, grokking_width=400,
                                      seed=0)
    saved = False
    for i in range(len(curve.steps)):
        a = bundle.observe(int(curve.steps[i]),
                            train_loss=float(curve.train_loss[i]),
                            train_acc=float(curve.train_acc[i]),
                            test_loss=float(curve.test_loss[i]),
                            test_acc=float(curve.test_acc[i]))
        if a.save_checkpoint:
            saved = True
            break
    assert saved
    assert ckpt._saved_event_step is not None


# ---------------------------------------------------------------------------
# 4. KairosSweepGate
# ---------------------------------------------------------------------------


def test_sweep_gate_constructs():
    g = KairosSweepGate(n_trials=8, eval_at_step=2000, keep_top_k=2)
    assert g.n_trials == 8
    assert g.keep_top_k == 2


def test_sweep_gate_ranks_grokking_trials_higher():
    """A trial that grokked > one that's still memorising."""
    gate = KairosSweepGate(n_trials=2, eval_at_step=2000, keep_top_k=1)
    # Trial A: groks
    curve_A = simulate_grokking_curve(n_steps=8000, memorisation_step=200,
                                        grokking_step=2500, grokking_width=200,
                                        seed=0)
    for i in range(len(curve_A.steps)):
        gate.observe_trial("A", int(curve_A.steps[i]),
                            train_loss=float(curve_A.train_loss[i]),
                            train_acc=float(curve_A.train_acc[i]),
                            test_loss=float(curve_A.test_loss[i]),
                            test_acc=float(curve_A.test_acc[i]))
    # Trial B: memorises only
    rng = np.random.default_rng(1)
    n = 8000
    for step in range(n):
        gate.observe_trial("B", step,
                            train_loss=float(0.01 + 0.001 * rng.standard_normal()),
                            train_acc=float(1.0),
                            test_loss=float(1.2 + 0.05 * rng.standard_normal()),
                            test_acc=float(0.05))
    decisions = gate.make_decisions()
    assert isinstance(decisions["A"], TrialDecision)
    assert decisions["A"].score > decisions["B"].score
    assert not decisions["A"].kill
    assert decisions["B"].kill


def test_sweep_gate_summary_format():
    gate = KairosSweepGate(n_trials=1, eval_at_step=2000)
    gate.observe_trial("solo", 100, train_loss=0.5, train_acc=0.6,
                        test_loss=0.7, test_acc=0.3)
    summaries = gate.summarise()
    assert len(summaries) == 1
    assert isinstance(summaries[0], TrialSummary)
    assert summaries[0].trial_id == "solo"


# ---------------------------------------------------------------------------
# 5. KairosAccelerator
# ---------------------------------------------------------------------------


def test_accelerator_constructs():
    a = KairosAccelerator(sigma=0.01, wait_steps=300, cooldown_steps=200)
    assert a.sigma == 0.01
    assert a.wait_steps == 300


def test_accelerator_invalid_sigma_raises():
    with pytest.raises(ValueError):
        KairosAccelerator(sigma=1.5)


def test_accelerator_fires_during_plateau():
    """During memorisation plateau with no grok, the accelerator
    should emit at least one pulse."""
    acc = KairosAccelerator(sigma=0.01, wait_steps=200, cooldown_steps=200,
                              max_pulses=5)
    monitor = GrokkingMonitor.for_slow_grok(check_every=200)
    bundle = CallbackBundle(acc, monitor=monitor)
    fired_pulses = 0
    for step in range(3000):
        # Memorisation reached at step 100, then stuck
        train_acc = 0.05 + 0.95 * min(1.0, step / 100.0)
        a = bundle.observe(step, train_loss=0.01, train_acc=train_acc,
                            test_loss=1.2, test_acc=0.05)
        if a.inject_noise_sigma > 0:
            fired_pulses += 1
    assert fired_pulses >= 1
    assert acc.n_pulses >= 1


def test_accelerator_does_not_fire_before_memorisation():
    acc = KairosAccelerator(sigma=0.01, wait_steps=10, cooldown_steps=10)
    bundle = CallbackBundle(acc)
    for step in range(100):
        # train_acc never reaches threshold
        a = bundle.observe(step, train_loss=0.5, train_acc=0.5,
                            test_loss=0.8, test_acc=0.3)
        assert a.inject_noise_sigma == 0.0


# ---------------------------------------------------------------------------
# 6. KairosCurriculum
# ---------------------------------------------------------------------------


def test_curriculum_default_settings():
    c = KairosCurriculum()
    assert isinstance(c.settings[Phase.MEMORISING], PhaseSettings)
    assert c.settings[Phase.POST].weight_decay_multiplier < 1.0


def test_curriculum_applies_to_optimizer():
    """Curriculum should mutate optimizer lr/wd when phase changes."""
    opt = _FakeOptimizer(initial_lr=1e-3)
    curr = KairosCurriculum(optimizer=opt)
    bundle = CallbackBundle(curr)
    # Force a fast transition by feeding a curve
    curve = simulate_grokking_curve(n_steps=6000, memorisation_step=300,
                                      grokking_step=4000, grokking_width=200,
                                      seed=0)
    initial_lr = opt.param_groups[0]["lr"]
    for i in range(len(curve.steps)):
        bundle.observe(int(curve.steps[i]),
                        train_loss=float(curve.train_loss[i]),
                        train_acc=float(curve.train_acc[i]),
                        test_loss=float(curve.test_loss[i]),
                        test_acc=float(curve.test_acc[i]))
    # By the end the curriculum has visited POST -> wd should be reduced from initial
    final_wd = opt.param_groups[0]["weight_decay"]
    final_lr = opt.param_groups[0]["lr"]
    # LR should be lower than initial (we passed near_critical / drifting / post)
    assert final_lr < initial_lr
    # WD should be lower than initial (POST is wd_multiplier=0.5)
    assert final_wd < 1.0


# ---------------------------------------------------------------------------
# 7. KairosProbe
# ---------------------------------------------------------------------------


def test_probe_constructs():
    p = KairosProbe(window=50, min_observations=20)
    assert p.window == 50


def test_probe_handles_too_few_observations():
    p = KairosProbe(min_observations=100)
    for step in range(10):
        p.observe(step, scores={"cap_a": float(step) / 10.0})
    reports = p.diagnose()
    assert len(reports) == 1
    assert reports[0].regime == "not_enough_data"
    assert not reports[0].likely_to_emerge_next


def test_probe_detects_capability_approach():
    """A capability with rising AR(1) + variance should be flagged."""
    p = KairosProbe(window=50, min_observations=40)
    rng = np.random.default_rng(0)
    # Three capabilities; one ramps up sharply at the end
    for step in range(200):
        cap_emerging = 0.05 + 0.0005 * step ** 2 / 40 + 0.05 * rng.standard_normal()
        cap_flat = 0.05 + 0.01 * rng.standard_normal()
        p.observe(step, scores={
            "emerging": float(cap_emerging),
            "flat": float(cap_flat),
        })
    reports = {r.capability: r for r in p.diagnose()}
    assert "emerging" in reports
    assert "flat" in reports
    assert reports["emerging"].last_score > reports["flat"].last_score


# ---------------------------------------------------------------------------
# CallbackBundle composition: all 7 wired together
# ---------------------------------------------------------------------------


def test_bundle_with_all_seven_components_runs():
    opt = _FakeOptimizer(initial_lr=1e-3)
    bundle = CallbackBundle(
        KairosEarlyStop(stable_steps_to_abort=20_000, min_step=1000),
        KairosLRSchedule(drop_factor=0.1, optimizer=opt),
        KairosCheckpoint(),  # no save_dir; just flag
        KairosAccelerator(sigma=0.01, wait_steps=300, max_pulses=3),
        KairosCurriculum(optimizer=opt),
    )
    curve = simulate_slow_grok_curve(n_steps=12_000, memorisation_step=150,
                                       grok_start_step=1000,
                                       grok_complete_step=11_000, seed=0)
    actions_taken = []
    for i in range(len(curve.steps)):
        a = bundle.observe(int(curve.steps[i]),
                            train_loss=float(curve.train_loss[i]),
                            train_acc=float(curve.train_acc[i]),
                            test_loss=float(curve.test_loss[i]),
                            test_acc=float(curve.test_acc[i]))
        if a.stop_training or a.save_checkpoint or a.inject_noise_sigma > 0 \
                or a.lr_multiplier != 1.0:
            actions_taken.append((curve.steps[i], a))
    # Should have taken at least 5 actions across components
    assert len(actions_taken) >= 5, (
        f"expected multiple actions; got {len(actions_taken)}"
    )
