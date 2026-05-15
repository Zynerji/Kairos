"""Grounding pool -- HotpotQA with context (RAG-style).

Target: answer supported by provided context or abstain if absent.
Score: F1 on generated answer vs. gold.
"""
from __future__ import annotations
from dataclasses import dataclass

from kairos.aletheia.pools.hf_base import HFCausalLMPool, normalized_f1


@dataclass
class GroundingPool(HFCausalLMPool):
    train_dataset_id: str = "hotpot_qa"
    train_subset: str | None = "distractor"
    eval_dataset_id: str = "hotpot_qa"
    eval_subset: str | None = "distractor"
    eval_split: str = "validation"
    max_len: int = 2048
    max_new_tokens: int = 96

    @property
    def name(self) -> str:
        return "grounding"

    def _flatten_context(self, ctx) -> str:
        if isinstance(ctx, dict):
            titles = ctx.get("title") or []
            sentences = ctx.get("sentences") or []
            parts = []
            for t, ss in zip(titles, sentences):
                body = " ".join(ss) if isinstance(ss, list) else str(ss)
                parts.append(f"[{t}] {body}")
            return "\n".join(parts)
        return str(ctx)

    def _format_example(self, ex: dict) -> tuple[str, str]:
        q = ex.get("question", "")
        a = ex.get("answer", "")
        ctx = self._flatten_context(ex.get("context", {}))
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        return prompt, f" {a}"

    def _score(self, prediction: str, gold: str) -> float:
        return normalized_f1(prediction.split("\n")[0], gold)
