"""Teacher self-consistency rejection sampling.

Keep teacher outputs only when N resamples agree above threshold.
Cheap hallucination filter that needs no ground-truth labels.

Usage:
    sampler = RejectionSampler(n_resamples=3, agreement_threshold=0.66)
    if sampler.keep(resampled_outputs):
        ingest(consensus)
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass


@dataclass
class RejectionSampler:
    n_resamples: int = 3
    agreement_threshold: float = 0.66
    normalize: bool = True                  # lowercase + strip whitespace

    def __post_init__(self) -> None:
        if not 0.0 < self.agreement_threshold <= 1.0:
            raise ValueError(f"agreement_threshold must be in (0,1], got {self.agreement_threshold}")
        if self.n_resamples < 1:
            raise ValueError("n_resamples must be >= 1")

    def _norm(self, s: str) -> str:
        return s.strip().lower() if self.normalize else s

    def _top(self, samples: list[str]) -> tuple[str, float]:
        if not samples:
            return "", 0.0
        normed = [self._norm(s) for s in samples]
        counts = Counter(normed)
        top, n = counts.most_common(1)[0]
        return top, n / len(samples)

    def keep(self, samples: list[str]) -> bool:
        _, frac = self._top(samples)
        return frac >= self.agreement_threshold

    def consensus(self, samples: list[str]) -> str | None:
        top, frac = self._top(samples)
        return top if frac >= self.agreement_threshold else None

    def agreement(self, samples: list[str]) -> float:
        return self._top(samples)[1]
