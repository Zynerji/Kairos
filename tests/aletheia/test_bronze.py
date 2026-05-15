import math
import pytest
from kairos.aletheia.torsion.bronze import BronzePendulum, BRONZE_RATIO, BRONZE_ANGLE, GOLDEN_ANGLE


def test_constants():
    # beta_3 = (3 + sqrt(13)) / 2, solves x^2 = 3x + 1
    assert abs(BRONZE_RATIO ** 2 - 3.0 * BRONZE_RATIO - 1.0) < 1e-10
    # angle ~= 109 deg
    assert 1.89 < BRONZE_ANGLE < 1.91
    # golden angle ~= 137.5 deg
    assert 2.39 < GOLDEN_ANGLE < 2.41


def test_phases_spread_by_golden_angle():
    heads = ["a", "b", "c", "d"]
    b = BronzePendulum(heads)
    # Each consecutive pair differs by GOLDEN_ANGLE mod 2pi
    for i in range(len(heads) - 1):
        diff = (b.phases[heads[i + 1]] - b.phases[heads[i]]) % (2 * math.pi)
        expected = GOLDEN_ANGLE % (2 * math.pi)
        assert abs(diff - expected) < 1e-9


def test_weights_clamped():
    b = BronzePendulum(["x"], amplitude=10.0, floor=0.3, ceil=2.0)
    for step in range(200):
        w = b.weight("x", step)
        assert 0.3 <= w <= 2.0


def test_weights_in_bounds_default():
    b = BronzePendulum(["a", "b"])
    for step in range(500):
        for h in ["a", "b"]:
            w = b.weight(h, step)
            assert 0.3 <= w <= 2.0


def test_base_weight_respected():
    b = BronzePendulum(["x"], amplitude=0.0, base_weights={"x": 1.5})
    # amp=0 means weight = base_weight * 1.0 = 1.5 always
    assert abs(b.weight("x", 0) - 1.5) < 1e-10
    assert abs(b.weight("x", 100) - 1.5) < 1e-10


def test_weights_dict_covers_all_heads():
    b = BronzePendulum(["a", "b", "c"])
    ws = b.weights(step=7)
    assert set(ws.keys()) == {"a", "b", "c"}


def test_unknown_head_raises():
    b = BronzePendulum(["a"])
    with pytest.raises(KeyError):
        b.weight("z", 0)


def test_empty_heads_rejected():
    with pytest.raises(ValueError):
        BronzePendulum([])


def test_invalid_clamp_rejected():
    with pytest.raises(ValueError):
        BronzePendulum(["a"], floor=2.0, ceil=1.0)


def test_anti_resonant_does_not_repeat_early():
    """Bronze angle is irrational -- weights should not repeat exactly for many steps."""
    b = BronzePendulum(["x"], amplitude=0.4, floor=0.01, ceil=10.0)
    ws = [b.weight("x", i) for i in range(50)]
    # No two consecutive steps produce identical weights to 6 decimals
    for i in range(len(ws) - 1):
        assert abs(ws[i] - ws[i + 1]) > 1e-6
