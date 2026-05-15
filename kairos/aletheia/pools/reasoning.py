"""Reasoning pool -- GSM8K primary; MATH / HumanEval via benchmark extension.

For thinking-model targets, target sequences include <think> blocks
when thinking_mode=True. Teacher distillation (R1-Distill) feeds this
pool directly via the distillation pool's Phase B role.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from kairos.aletheia.pools.hf_base import HFCausalLMPool


@dataclass
class ReasoningPool(HFCausalLMPool):
    train_dataset_id: str = "gsm8k"
    train_subset: str | None = "main"
    eval_dataset_id: str = "gsm8k"
    eval_subset: str | None = "main"
    eval_split: str = "test"
    max_len: int = 1024
    max_new_tokens: int = 256
    thinking_mode: bool = True

    @property
    def name(self) -> str:
        return "reasoning"

    def _format_example(self, ex: dict) -> tuple[str, str]:
        q = ex.get("question") or ex.get("problem") or ""
        a = ex.get("answer") or ex.get("solution") or ""
        if self.thinking_mode:
            prompt = f"Problem: {q}\n<think>\n"
            target = f"{a}"
        else:
            prompt = f"Problem: {q}\nSolution:"
            target = f" {a}"
        return prompt, target

    def _extract_number(self, text: str) -> str | None:
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if m:
            return m.group(1)
        # fallback: last number in last 100 chars
        nums = re.findall(r"(-?\d+(?:\.\d+)?)", text[-100:])
        return nums[-1] if nums else None

    def _score(self, prediction: str, gold: str) -> float:
        p = self._extract_number(prediction)
        g = self._extract_number(gold)
        if p is None or g is None:
            return 0.0
        try:
            return float(abs(float(p) - float(g)) < 1e-6)
        except ValueError:
            return 0.0
