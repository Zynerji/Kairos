"""Pool abstract base class + common CausalLM pool helper.

A Pool bundles:
- Empirical target dataset (train split)
- Out-of-distribution eval set (OOT split, strict no-leak)
- Loss function for Phase A training
- OOT evaluation metric returning a score in [0, 1] (higher = better)
- Optional teacher distillation source

Each pool owns its LoRA adapter key. The TorsionCycle activates one
pool's adapter at a time in Phase A, running a sub-block whose length
is modulated by the Torus pendulum and per-step loss weight by Bronze.

Subclass contract:
- name: str                          (unique; used by ratchet/pendulum)
- train_loader(bs) -> Iterator       (Phase A training batches)
- oot_loader(bs)   -> Iterator       (evaluation batches, DISJOINT)
- loss(batch, model, scale) -> Tensor (Phase A scalar loss)
- evaluate(model)  -> PoolEvalResult (OOT; score higher-is-better)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Iterable


@dataclass
class PoolBatch:
    """Generic batch container passed to pool loss functions."""
    inputs: Any                           # tokenized inputs (usually dict for HF)
    targets: Any                          # pool-specific target tensor(s)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolEvalResult:
    name: str
    score: float                          # in [0, 1], higher is better
    components: dict[str, float] = field(default_factory=dict)
    n_examples: int = 0


class Pool(ABC):
    """Abstract pool interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def adapter_name(self) -> str:
        return f"aletheia_{self.name}"

    @abstractmethod
    def train_loader(self, batch_size: int) -> Iterator[PoolBatch]: ...

    @abstractmethod
    def oot_loader(self, batch_size: int) -> Iterator[PoolBatch]: ...

    @abstractmethod
    def loss(self, batch: PoolBatch, model: Any, scale: float = 1.0) -> Any: ...

    @abstractmethod
    def evaluate(self, model: Any, batch_size: int = 8) -> PoolEvalResult: ...

    def distill_batches(self, batch_size: int) -> Iterable[PoolBatch] | None:
        """Optional teacher-distilled supervision. Default: none."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class CausalLMPool(Pool):
    """Common base for pools whose Phase A loss is CE on target tokens.

    Subclasses typically just implement train_loader/oot_loader/evaluate,
    and optionally override loss() for non-CE objectives (calibration
    uses Brier, consistency uses pairwise, etc).

    Expects PoolBatch.inputs to be a dict compatible with model(**inputs)
    and PoolBatch.targets to be a LongTensor of labels with -100 on
    positions to ignore.
    """

    def loss(self, batch: PoolBatch, model: Any, scale: float = 1.0) -> Any:
        import torch.nn.functional as F

        out = model(**batch.inputs)
        logits = out.logits
        # [B, T, V] -> [B*T, V], [B, T] -> [B*T]
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.targets.view(-1),
            ignore_index=-100,
        )
        return scale * ce
