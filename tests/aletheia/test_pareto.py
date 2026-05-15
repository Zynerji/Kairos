import pytest
from pathlib import Path
from kairos.aletheia.ratchet.pareto import ParetoRatchet


def test_init_sets_best_from_anchor():
    anchor = {"fact": 0.5, "cal": 0.4}
    r = ParetoRatchet(anchor=anchor)
    assert r.best_scores == anchor
    assert r.best_product > 0


def test_product_formula():
    r = ParetoRatchet(anchor={"a": 0.5, "b": 0.4})
    assert abs(r.product({"a": 0.6, "b": 0.5}) - 0.30) < 1e-10


def test_product_eps_floor():
    r = ParetoRatchet(anchor={"a": 0.5, "b": 0.4}, eps=1e-3)
    # Zero on axis b -> clamp to eps
    p = r.product({"a": 0.6, "b": 0.0})
    assert abs(p - 0.6 * 1e-3) < 1e-10


def test_below_floor_axes_identifies_correctly():
    r = ParetoRatchet(anchor={"a": 1.0, "b": 1.0, "c": 1.0}, floor=0.8)
    scores = {"a": 0.9, "b": 0.7, "c": 0.5}
    below = r.below_floor_axes(scores)
    assert set(below) == {"b", "c"}


def test_single_axis_dip_no_rollback():
    r = ParetoRatchet(anchor={"a": 1.0, "b": 1.0, "c": 1.0}, floor=0.8)
    scores = {"a": 0.9, "b": 0.9, "c": 0.5}  # only c below floor
    assert not r.should_rollback(scores)


def test_dual_regression_triggers_rollback():
    r = ParetoRatchet(anchor={"a": 1.0, "b": 1.0, "c": 1.0}, floor=0.8)
    scores = {"a": 0.9, "b": 0.5, "c": 0.5}  # b and c below floor
    assert r.should_rollback(scores)


def test_new_best_requires_no_floor_violations():
    r = ParetoRatchet(anchor={"a": 0.5, "b": 0.5}, floor=0.8)
    # Product way higher but axis a below floor
    scores = {"a": 0.3, "b": 5.0}
    assert not r.is_new_best(scores)


def test_new_best_on_improvement():
    r = ParetoRatchet(anchor={"a": 0.5, "b": 0.5}, floor=0.8)
    scores = {"a": 0.6, "b": 0.6}
    assert r.is_new_best(scores)


def test_update_persists_best():
    r = ParetoRatchet(anchor={"a": 0.5, "b": 0.5})
    new_scores = {"a": 0.7, "b": 0.7}
    ckpt = Path("/tmp/ckpt.pt")
    r.update(new_scores, ckpt)
    assert r.best_scores == new_scores
    assert r.best_checkpoint == ckpt
    # Subsequent equal is not "new best"
    assert not r.is_new_best(new_scores)


def test_empty_anchor_rejected():
    with pytest.raises(ValueError):
        ParetoRatchet(anchor={})


def test_invalid_floor_rejected():
    with pytest.raises(ValueError):
        ParetoRatchet(anchor={"a": 1.0}, floor=0.0)
    with pytest.raises(ValueError):
        ParetoRatchet(anchor={"a": 1.0}, floor=1.5)
