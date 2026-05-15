"""Factuality pool -- TriviaQA (train) + SimpleQA (eval).

Target: short-answer QA. Normalized F1 against gold answer.
Anti-hallucination core pool.
"""
from __future__ import annotations
from dataclasses import dataclass

from kairos.aletheia.pools.hf_base import HFCausalLMPool, normalized_f1


@dataclass
class FactualityPool(HFCausalLMPool):
    train_dataset_id: str = "mandarjoshi/trivia_qa"
    train_subset: str | None = "rc.wikipedia.nocontext"
    eval_dataset_id: str = "basicv8/SimpleQA"
    eval_split: str = "test"
    max_len: int = 512
    max_new_tokens: int = 64

    @property
    def name(self) -> str:
        return "factuality"

    def _format_example(self, ex: dict) -> tuple[str, str]:
        q = ex.get("question") or ex.get("problem") or ""
        a_raw = ex.get("answer") or ex.get("answers") or ""
        if isinstance(a_raw, dict):
            a = a_raw.get("value") or (a_raw.get("aliases") or [""])[0]
        elif isinstance(a_raw, list):
            a = a_raw[0] if a_raw else ""
        else:
            a = str(a_raw)
        return f"Question: {q}\nAnswer:", f" {a}"

    def _score(self, prediction: str, gold: str) -> float:
        first_line = prediction.split("\n")[0]
        return normalized_f1(first_line, gold)
