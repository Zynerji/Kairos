import pytest
from kairos.aletheia.pools.base import Pool, CausalLMPool, PoolBatch, PoolEvalResult


def test_pool_is_abstract():
    with pytest.raises(TypeError):
        Pool()


class _ConcretePool(Pool):
    @property
    def name(self):
        return "concrete"

    def train_loader(self, batch_size):
        yield PoolBatch(inputs=None, targets=None)

    def oot_loader(self, batch_size):
        yield PoolBatch(inputs=None, targets=None)

    def loss(self, batch, model, scale=1.0):
        return scale

    def evaluate(self, model, batch_size=8):
        return PoolEvalResult(self.name, 0.5, {}, 1)


def test_concrete_pool_instantiates():
    p = _ConcretePool()
    assert p.name == "concrete"


def test_adapter_name_default_prefix():
    p = _ConcretePool()
    assert p.adapter_name == "aletheia_concrete"


def test_evaluate_returns_result():
    p = _ConcretePool()
    r = p.evaluate(model=None)
    assert isinstance(r, PoolEvalResult)
    assert r.name == "concrete"
    assert 0.0 <= r.score <= 1.0


def test_distill_batches_default_none():
    p = _ConcretePool()
    assert p.distill_batches(batch_size=4) is None


def test_pool_repr_has_name():
    p = _ConcretePool()
    assert "concrete" in repr(p)


def test_pool_batch_defaults():
    b = PoolBatch(inputs=[1, 2], targets=[3, 4])
    assert b.metadata == {}


def test_eval_result_defaults():
    r = PoolEvalResult(name="x", score=0.7)
    assert r.components == {}
    assert r.n_examples == 0


def test_causallm_pool_is_abstract():
    # CausalLMPool still abstract (missing loaders/evaluate/name)
    with pytest.raises(TypeError):
        CausalLMPool()
