"""Abstention pool -- epistemic 'I don't know'.

For ABLATED models this is PURELY epistemic. We synthesize unanswerable
examples by hash-deterministic perturbation of real QA items (replacing
key entities with invented tokens). Answerable twins remain to avoid
over-abstention.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any

from kairos.aletheia.pools.base import PoolEvalResult
from kairos.aletheia.pools.hf_base import HFCausalLMPool


INVENTED_ENTITY_POOL = (
    "Zxqwpol Kjrvnb", "Mpvotrix Glandarian", "Qbuflen",
    "the Voskrian Treaty", "Hemarath Plateau", "the Xylorine Accord",
)
ABSTAIN_PHRASES = (
    "don't know", "cannot determine", "not sure", "unable",
    "insufficient information", "not enough context",
)


@dataclass
class AbstentionPool(HFCausalLMPool):
    train_dataset_id: str = "mandarjoshi/trivia_qa"
    train_subset: str | None = "rc.wikipedia.nocontext"
    eval_dataset_id: str = "mandarjoshi/trivia_qa"
    eval_subset: str | None = "rc.wikipedia.nocontext"
    eval_split: str = "validation"
    unanswerable_fraction: float = 0.5

    @property
    def name(self) -> str:
        return "abstention"

    def _is_unanswerable(self, ex: dict) -> bool:
        q = ex.get("question", "")
        h = hashlib.sha256(q.encode("utf-8", errors="replace")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF < self.unanswerable_fraction

    def _perturb(self, q: str) -> str:
        entity = INVENTED_ENTITY_POOL[hash(q) % len(INVENTED_ENTITY_POOL)]
        words = q.split()
        for i, w in enumerate(words):
            if w[:1].isupper() and len(w) > 2:
                words[i] = entity
                return " ".join(words)
        return f"What is the significance of {entity}?"

    def _format_example(self, ex: dict) -> tuple[str, str]:
        q = ex.get("question", "")
        a_raw = ex.get("answer", "")
        a = a_raw.get("value") if isinstance(a_raw, dict) else a_raw
        if self._is_unanswerable(ex):
            return f"Question: {self._perturb(q)}\nAnswer:", " I don't know."
        return f"Question: {q}\nAnswer:", f" {a}"

    def _score(self, prediction: str, gold: str) -> float:
        predicted_abstain = any(p in prediction.lower() for p in ABSTAIN_PHRASES)
        gold_abstain = "don't know" in gold.lower()
        return float(predicted_abstain == gold_abstain)
