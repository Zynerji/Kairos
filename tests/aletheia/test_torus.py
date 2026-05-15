import math
import pytest
from kairos.aletheia.torsion.torus import (
    TorusPendulum, PHI, PHI2, THETA1_STEP, THETA2_STEP, GOLDEN_ANGLE
)


def test_phi_identities():
    assert abs(PHI ** 2 - PHI - 1.0) < 1e-10
    assert abs(PHI2 - (PHI + 1.0)) < 1e-10
    # theta1 = 2*pi / phi^2 ~= 137.5 deg (golden angle)
    assert abs(THETA1_STEP - GOLDEN_ANGLE) < 1e-9


def test_theta_rates_distinct():
    # Golden (137.5) and bronze (109) must be distinct for quasiperiodicity
    assert abs(THETA1_STEP - THETA2_STEP) > 0.3


def test_weight_clamped():
    t = TorusPendulum(["a"], weight_amplitude=10.0, floor=0.3, ceil=2.0)
    for step in range(200):
        w = t.weight("a", step)
        assert 0.3 <= w <= 2.0


def test_step_count_positive():
    t = TorusPendulum(["a"], step_amplitude=5.0, base_steps=10)
    for step in range(200):
        assert t.step_count("a", step) >= 1


def test_step_count_default_range():
    t = TorusPendulum(["a"], step_amplitude=0.3, base_steps=100)
    counts = [t.step_count("a", s) for s in range(200)]
    # amp 0.3 -> modulation in [0.7, 1.3] -> counts in [70, 130]
    assert min(counts) >= 70
    assert max(counts) <= 130


def test_never_revisits_weight_schedule_pair():
    """Golden x Bronze is irrational -- no (weight, step_count) pair repeats exactly."""
    t = TorusPendulum(["x"], base_steps=100)
    pairs = set()
    for step in range(300):
        key = (round(t.weight("x", step), 6), t.step_count("x", step))
        pairs.add(key)
    # With irrational frequencies, we expect ~300 distinct pairs
    assert len(pairs) > 250


def test_phases_spread_by_golden_angle():
    heads = ["a", "b", "c"]
    t = TorusPendulum(heads)
    for i in range(len(heads) - 1):
        diff = (t.phases[heads[i + 1]] - t.phases[heads[i]]) % (2 * math.pi)
        expected = GOLDEN_ANGLE % (2 * math.pi)
        assert abs(diff - expected) < 1e-9


def test_unknown_head_raises():
    t = TorusPendulum(["a"])
    with pytest.raises(KeyError):
        t.weight("z", 0)
    with pytest.raises(KeyError):
        t.step_count("z", 0)


def test_weights_and_step_counts_cover_all_heads():
    t = TorusPendulum(["a", "b", "c"])
    assert set(t.weights(5).keys()) == {"a", "b", "c"}
    assert set(t.step_counts(5).keys()) == {"a", "b", "c"}


def test_invalid_base_steps_rejected():
    with pytest.raises(ValueError):
        TorusPendulum(["a"], base_steps=0)
