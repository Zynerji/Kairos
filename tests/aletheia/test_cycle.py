"""TorsionCycle integration tests with mock pools and hooks (no torch)."""
from pathlib import Path
import tempfile
import pytest

from kairos.aletheia.torsion.cycle import TorsionCycle
from kairos.aletheia.torsion.bronze import BronzePendulum
from kairos.aletheia.torsion.torus import TorusPendulum
from kairos.aletheia.ratchet.pareto import ParetoRatchet
from kairos.aletheia.pools.base import Pool, PoolBatch, PoolEvalResult


class _FakeLoss:
    def __init__(self, v: float):
        self.v = float(v)
        self.backward_called = False

    def backward(self):
        self.backward_called = True

    def detach(self):
        return self

    def __float__(self):
        return self.v


class _FakePool(Pool):
    def __init__(self, name: str, score_trajectory: list[float]):
        self._name = name
        self.scores = list(score_trajectory)
        self.train_calls = 0
        self.eval_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def train_loader(self, batch_size):
        while True:
            self.train_calls += 1
            yield PoolBatch(inputs=None, targets=None)

    def oot_loader(self, batch_size):
        yield PoolBatch(inputs=None, targets=None)

    def loss(self, batch, model, scale=1.0):
        return _FakeLoss(scale * 0.5)

    def evaluate(self, model, batch_size=8):
        self.eval_calls += 1
        s = self.scores.pop(0) if self.scores else 0.5
        return PoolEvalResult(self._name, s, {}, 10)


def _noop(*args, **kwargs):
    return None


def _save_ckpt_stub(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake-ckpt")
    return path


def _make_cycle(pools, anchor, base_steps=3, phase_b_steps=2, batch_size=1):
    names = [p.name for p in pools]
    return TorsionCycle(
        pools=pools,
        ratchet=ParetoRatchet(anchor=anchor, floor=0.8),
        bronze=BronzePendulum(heads=names),
        torus=TorusPendulum(heads=names, base_steps=base_steps),
        phase_b_steps=phase_b_steps,
        batch_size=batch_size,
        activate_pool_adapter=_noop,
        freeze_backbone=_noop,
        unfreeze_backbone=_noop,
        freeze_adapters=_noop,
        unfreeze_adapters=_noop,
        phase_b_step=lambda m, s: 0.1,
        save_checkpoint=_save_ckpt_stub,
        restore_checkpoint=_noop,
        optimizer_step=_noop,
    )


def test_cycle_runs_end_to_end():
    pools = [_FakePool("a", [0.6, 0.7]), _FakePool("b", [0.6, 0.7])]
    with tempfile.TemporaryDirectory() as d:
        cycle = _make_cycle(pools, {"a": 0.5, "b": 0.5})
        state = cycle.run(model=None, max_cycles=2, output_dir=Path(d))
    assert state.cycle == 1
    assert set(state.last_scores.keys()) == {"a", "b"}


def test_cycle_records_new_best_on_improvement():
    # Anchor 0.5/0.5; new scores 0.7/0.7 > product and above floor -> new best
    pools = [_FakePool("a", [0.7]), _FakePool("b", [0.7])]
    with tempfile.TemporaryDirectory() as d:
        cycle = _make_cycle(pools, {"a": 0.5, "b": 0.5})
        state = cycle.run(model=None, max_cycles=1, output_dir=Path(d))
    assert state.new_bests == 1
    assert cycle.ratchet.best_checkpoint is not None


def test_cycle_triggers_rollback_on_dual_regression():
    # Cycle 0 improves (0.8 > anchor 0.5) -> new_best, saves checkpoint.
    # Cycle 1 drops both axes to 0.3 < floor*anchor (0.8*0.5=0.4) -> rollback.
    pools = [_FakePool("a", [0.8, 0.3]), _FakePool("b", [0.8, 0.3])]
    with tempfile.TemporaryDirectory() as d:
        cycle = _make_cycle(pools, {"a": 0.5, "b": 0.5})
        state = cycle.run(model=None, max_cycles=2, output_dir=Path(d))
    assert state.new_bests == 1
    assert state.rollbacks == 1


def test_cycle_duplicate_pool_names_rejected():
    pools = [_FakePool("x", [0.5]), _FakePool("x", [0.5])]
    with pytest.raises(ValueError, match="unique"):
        _make_cycle(pools, {"x": 0.5})


def test_cycle_missing_hook_rejected():
    pools = [_FakePool("a", [0.5])]
    with pytest.raises(ValueError, match="missing hooks"):
        TorsionCycle(
            pools=pools,
            ratchet=ParetoRatchet(anchor={"a": 0.5}),
            bronze=BronzePendulum(heads=["a"]),
            torus=TorusPendulum(heads=["a"]),
            # no hooks supplied
        )


def test_cycle_phase_a_trains_every_pool():
    pools = [_FakePool("a", [0.5, 0.5]), _FakePool("b", [0.5, 0.5])]
    with tempfile.TemporaryDirectory() as d:
        cycle = _make_cycle(pools, {"a": 0.5, "b": 0.5}, base_steps=3)
        cycle.run(model=None, max_cycles=1, output_dir=Path(d))
    assert pools[0].train_calls > 0
    assert pools[1].train_calls > 0


def test_cycle_evaluates_every_pool_per_cycle():
    pools = [_FakePool("a", [0.5, 0.6]), _FakePool("b", [0.5, 0.6])]
    with tempfile.TemporaryDirectory() as d:
        cycle = _make_cycle(pools, {"a": 0.5, "b": 0.5})
        cycle.run(model=None, max_cycles=2, output_dir=Path(d))
    assert pools[0].eval_calls == 2
    assert pools[1].eval_calls == 2


def test_log_cb_receives_records():
    pools = [_FakePool("a", [0.6, 0.7])]
    records = []
    with tempfile.TemporaryDirectory() as d:
        cycle = _make_cycle(pools, {"a": 0.5})
        cycle.run(model=None, max_cycles=2, output_dir=Path(d), log_cb=records.append)
    assert len(records) == 2
    assert {"cycle", "scores", "event", "elapsed_s", "best_product"}.issubset(records[0])
