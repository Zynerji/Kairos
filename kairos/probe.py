"""KairosProbe: generic capability-emergence probe.

Beyond grokking specifically, modern large-model training is full of
"phase transitions" — emergent capabilities that appear sharply at
some training scale or step. Examples: in-context learning emergence,
chain-of-thought emergence, multi-hop reasoning emergence.

The probe takes a list of capability-test scores (one per step) and
runs Cassandra over each, reporting:
  * regime per capability
  * which capabilities are showing CSD precursors right now
  * predicted emergence ordering (which will emerge next)

This is the most experimental of the seven components. The Cassandra
signature on test-score streams may or may not be a robust early-
warning indicator for capability emergence at scale; it's a hypothesis
worth probing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from grokking_monitor import _cassandra_path  # noqa: F401 -- side-effect
from cassandra import diagnose as _cassandra_diagnose


@dataclass(frozen=True)
class EmergenceReport:
    """Per-capability emergence verdict.

    Attributes
    ----------
    capability : str
        Name of the capability being probed.
    regime : str
        Cassandra regime: stable / drifting / near_critical / post.
    ar1_trend_tau : float
        Kendall's tau of the AR(1) trend on this capability's score
        history. Positive = approaching transition.
    var_trend_tau : float
        Same for variance.
    last_score : float
    likely_to_emerge_next : bool
        True iff the regime is `near_critical` or both taus are large.
    notes : str
    """
    capability: str
    regime: str
    ar1_trend_tau: float
    var_trend_tau: float
    last_score: float
    likely_to_emerge_next: bool
    notes: str


class KairosProbe:
    """Probe multiple capability scores in parallel.

    Usage:
        probe = KairosProbe()
        # call once per evaluation pass
        for step in eval_steps:
            scores = {"icl": icl_score, "cot": cot_score, ...}
            probe.observe(step, scores=scores)
        report = probe.diagnose()
        for r in report:
            print(r.capability, r.regime, r.likely_to_emerge_next)
    """

    def __init__(self, window: int = 50, min_observations: int = 20) -> None:
        self.window = int(window)
        self.min_observations = int(min_observations)
        self._scores: dict[str, list[float]] = defaultdict(list)
        self._steps: list[int] = []

    def observe(self, step: int, scores: dict[str, float]) -> None:
        self._steps.append(int(step))
        for k, v in scores.items():
            self._scores[str(k)].append(float(v))

    def diagnose(self) -> list[EmergenceReport]:
        """Run Cassandra on each capability and aggregate the verdicts."""
        out: list[EmergenceReport] = []
        for cap, series in self._scores.items():
            if len(series) < self.min_observations:
                out.append(EmergenceReport(
                    capability=cap, regime="not_enough_data",
                    ar1_trend_tau=0.0, var_trend_tau=0.0,
                    last_score=float(series[-1] if series else float("nan")),
                    likely_to_emerge_next=False,
                    notes=f"need {self.min_observations} samples, have {len(series)}",
                ))
                continue
            arr = np.asarray(series, dtype=float)
            try:
                rep = _cassandra_diagnose(
                    arr, window=min(self.window, len(arr) // 3), step=1,
                )
            except Exception as e:
                out.append(EmergenceReport(
                    capability=cap, regime="error",
                    ar1_trend_tau=0.0, var_trend_tau=0.0,
                    last_score=float(arr[-1]),
                    likely_to_emerge_next=False,
                    notes=f"diagnose error: {type(e).__name__}: {e}",
                ))
                continue
            likely = (
                rep.regime == "near_critical"
                or (rep.ar1_trend_tau > 0.4 and rep.var_trend_tau > 0.4)
            )
            out.append(EmergenceReport(
                capability=cap, regime=rep.regime,
                ar1_trend_tau=float(rep.ar1_trend_tau),
                var_trend_tau=float(rep.var_trend_tau),
                last_score=float(arr[-1]),
                likely_to_emerge_next=bool(likely),
                notes=rep.notes,
            ))
        return out

    def emerging_now(self) -> list[str]:
        """Names of capabilities currently showing near-emergence signatures."""
        return [r.capability for r in self.diagnose() if r.likely_to_emerge_next]
