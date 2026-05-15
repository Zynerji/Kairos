"""Teacher output refusal filter.

Ablated-model-CRITICAL: every 'I cannot help with that' in teacher
output re-trains refusal into the student. Reject entire example if
teacher refused. Regex + optional classifier.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable


DEFAULT_REFUSAL_PATTERNS: list[str] = [
    r"\bI (can'?t|cannot|am unable to|am not able to|won'?t)\b",
    r"\b(As an AI|As a language model|As an assistant)\b",
    r"\bI (do not|don'?t) (have|possess) (the ability|personal)",
    r"\bI'?m (unable|not able) to (help|assist|provide)",
    r"\b(Sorry|Unfortunately)[^.]*\b(can'?t|cannot|not able)\b",
    r"\bI must (decline|refuse)",
    r"\bI apologize[^.]*\b(can'?t|cannot)\b",
    r"\b(against|violates?) (my|the) (programming|guidelines|policy|terms)",
    r"\bnot (appropriate|suitable|able) to (generate|provide|create)",
    r"\bI'?m sorry,? but I (can'?t|cannot)",
    r"\bIt'?s not appropriate",
    r"\bI'?m designed to (decline|refuse|avoid)",
]


@dataclass
class RefusalFilter:
    patterns: list[str] = field(default_factory=lambda: list(DEFAULT_REFUSAL_PATTERNS))
    classifier: Callable[[str], float] | None = None
    classifier_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def is_refusal(self, text: str) -> bool:
        for rx in self._compiled:
            if rx.search(text):
                return True
        if self.classifier is not None:
            return float(self.classifier(text)) >= self.classifier_threshold
        return False

    def filter_batch(self, texts: list[str]) -> tuple[list[int], list[int]]:
        """Return (keep_indices, reject_indices)."""
        keep, reject = [], []
        for i, t in enumerate(texts):
            (reject if self.is_refusal(t) else keep).append(i)
        return keep, reject
