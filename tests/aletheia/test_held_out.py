import pytest
from kairos.aletheia.eval.held_out import HashSplit, content_hash, assert_disjoint


def test_content_hash_deterministic():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_hash_split_deterministic():
    s = HashSplit(eval_fraction=0.1)
    # Same text always gets same assignment
    a = [s.assign(f"item-{i}") for i in range(50)]
    b = [s.assign(f"item-{i}") for i in range(50)]
    assert a == b


def test_hash_split_fraction_approximate():
    s = HashSplit(eval_fraction=0.2, salt="test")
    items = [f"item-{i}" for i in range(1000)]
    train, eval_ = s.split(items)
    frac = len(eval_) / len(items)
    # Allow ±25% tolerance on a 1000-sample stat
    assert 0.15 < frac < 0.25


def test_hash_split_partitions_exactly():
    s = HashSplit()
    items = [f"x-{i}" for i in range(100)]
    train, eval_ = s.split(items)
    assert len(set(train) & set(eval_)) == 0
    assert len(train) + len(eval_) == 100


def test_hash_split_salt_changes_assignment():
    a = HashSplit(salt="A")
    b = HashSplit(salt="B")
    # Different salts -> different (but each deterministic) assignment
    items = [f"t-{i}" for i in range(50)]
    assignments_a = [a.assign(x) for x in items]
    assignments_b = [b.assign(x) for x in items]
    assert assignments_a != assignments_b


def test_assert_disjoint_happy_path():
    train = ["a", "b", "c"]
    eval_ = ["d", "e"]
    assert_disjoint(train, eval_)  # no raise


def test_assert_disjoint_detects_leak():
    train = ["a", "b", "c"]
    eval_ = ["c", "d"]
    with pytest.raises(ValueError, match="leaks"):
        assert_disjoint(train, eval_)


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        HashSplit(eval_fraction=0.0)
    with pytest.raises(ValueError):
        HashSplit(eval_fraction=1.0)
