"""Strict train/eval split guard -- no leak.

jDHART v22->v23 lesson: in-domain val lies (IC=0.77 val vs OOT=-0.03).
ALWAYS assert split disjointness before training, and prefer hash-based
deterministic splits over random shuffles that are hard to reproduce.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Iterable


def content_hash(text: str, salt: str = "") -> str:
    return hashlib.sha256((salt + text).encode("utf-8", errors="replace")).hexdigest()


@dataclass
class HashSplit:
    """Deterministic hash-based train/eval split.

    Given the same text + salt, always assigns the same split. Safe
    across machines / runs / dataset shuffles.
    """
    eval_fraction: float = 0.1
    salt: str = "aletheia-v1"

    def __post_init__(self) -> None:
        if not 0.0 < self.eval_fraction < 1.0:
            raise ValueError(f"eval_fraction must be in (0,1), got {self.eval_fraction}")

    def assign(self, text: str) -> str:
        h = content_hash(text, self.salt)
        v = int(h[:8], 16) / 0xFFFFFFFF
        return "eval" if v < self.eval_fraction else "train"

    def split(self, examples: Iterable[str]) -> tuple[list[int], list[int]]:
        train_idx: list[int] = []
        eval_idx: list[int] = []
        for i, x in enumerate(examples):
            (eval_idx if self.assign(x) == "eval" else train_idx).append(i)
        return train_idx, eval_idx


def assert_disjoint(train_texts: Iterable[str], eval_texts: Iterable[str]) -> None:
    """Raise if any eval text appears in train (exact match after hashing)."""
    train_hashes = {content_hash(t) for t in train_texts}
    for t in eval_texts:
        if content_hash(t) in train_hashes:
            raise ValueError(f"eval example leaks into train: {t[:80]!r}")
