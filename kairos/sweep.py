"""KairosSweepGate: hyperparameter-sweep compute allocator.

Runs N trials briefly and uses Cassandra's CSD signature on each
trial's `test_loss` to decide which ones are showing promise. Kills
the rest. The remaining "promising" trials get the full compute
budget.

Usage pattern (caller-managed; this class only does decisioning):

    gate = KairosSweepGate(n_trials=8, eval_at_step=2000)
    for step, batch in shared_loop_over_all_trials:
        for trial_id in active_trials:
            metrics = train_one_step(trial_id, batch)
            gate.observe_trial(trial_id, step, **metrics)
        if step == 2000:
            decisions = gate.make_decisions()
            active_trials = [t for t in active_trials
                              if not decisions[t.id].kill]
"""

from __future__ import annotations

from dataclasses import dataclass

from grokking_monitor import GrokkingMonitor

from .core import BaseCallback


@dataclass(frozen=True)
class TrialSummary:
    """Per-trial summary at the decision point."""
    trial_id: str
    step: int
    monitor_state: dict
    last_test_loss: float
    last_test_acc: float
    last_train_acc: float
    grokking_detected: bool


@dataclass(frozen=True)
class TrialDecision:
    """Outcome of the gate for one trial."""
    trial_id: str
    kill: bool
    reason: str
    score: float                  # higher = more promising


class KairosSweepGate:
    """Multi-trial compute allocator.

    Parameters
    ----------
    n_trials : int
        Total number of trials being run.
    eval_at_step : int
        The step at which the gate's decision logic is applied. After
        this step, ``make_decisions()`` returns a verdict per trial.
    keep_top_k : int | None
        Number of trials to keep alive past `eval_at_step`. If None,
        defaults to ``max(1, n_trials // 4)``.
    monitor_factory : callable, optional
        Callable returning a new `GrokkingMonitor`. Defaults to
        ``GrokkingMonitor.for_slow_grok``.
    """

    def __init__(self, n_trials: int, eval_at_step: int = 2000,
                 keep_top_k: int | None = None,
                 monitor_factory=None) -> None:
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; got {n_trials}")
        if eval_at_step < 100:
            raise ValueError(f"eval_at_step too small ({eval_at_step})")
        self.n_trials = int(n_trials)
        self.eval_at_step = int(eval_at_step)
        self.keep_top_k = (
            int(keep_top_k) if keep_top_k is not None
            else max(1, self.n_trials // 4)
        )
        self.monitor_factory = monitor_factory or (
            lambda: GrokkingMonitor.for_slow_grok(check_every=200)
        )
        self._monitors: dict[str, GrokkingMonitor] = {}
        self._last_metrics: dict[str, dict] = {}

    def _get_monitor(self, trial_id: str) -> GrokkingMonitor:
        if trial_id not in self._monitors:
            self._monitors[trial_id] = self.monitor_factory()
        return self._monitors[trial_id]

    def observe_trial(self, trial_id: str, step: int, **metrics: float
                       ) -> None:
        """Feed one step of one trial's metrics into the gate."""
        m = self._get_monitor(str(trial_id))
        m.observe(step, **{
            k: v for k, v in metrics.items()
            if k in {"train_loss", "train_acc", "test_loss", "test_acc"}
        })
        self._last_metrics[str(trial_id)] = dict(metrics)

    def trial_score(self, trial_id: str) -> float:
        """Heuristic score: higher = more promising.

        Combines four signals:
          + grokking_event_detected (large bonus)
          + Cassandra ar1_trend_tau on test_loss (the CSD indicator)
          + Cassandra var_trend_tau on test_loss
          + (current test_acc - 0.05)  -- direct progress
        """
        m = self._get_monitor(str(trial_id))
        if m.detected_event is not None:
            return 100.0
        state = m.state()
        diag = state.last_cassandra_diagnosis
        ar1 = float(diag.get("ar1_trend_tau", 0.0) or 0.0) if diag else 0.0
        var = float(diag.get("var_trend_tau", 0.0) or 0.0) if diag else 0.0
        last = self._last_metrics.get(str(trial_id), {})
        test_acc = float(last.get("test_acc", 0.0))
        return max(0.0, ar1) + max(0.0, var) + max(0.0, test_acc - 0.05) * 2

    def summarise(self) -> list[TrialSummary]:
        out: list[TrialSummary] = []
        for tid, m in self._monitors.items():
            state = m.state()
            last = self._last_metrics.get(tid, {})
            out.append(TrialSummary(
                trial_id=tid,
                step=state.last_step,
                monitor_state=dict(state.last_cassandra_diagnosis or {}),
                last_test_loss=float(last.get("test_loss", float("nan"))),
                last_test_acc=float(last.get("test_acc", float("nan"))),
                last_train_acc=float(last.get("train_acc", float("nan"))),
                grokking_detected=(m.detected_event is not None),
            ))
        return out

    def make_decisions(self) -> dict[str, TrialDecision]:
        """Rank trials by score; kill all below the top-`keep_top_k`."""
        ranked = sorted(
            ((tid, self.trial_score(tid)) for tid in self._monitors),
            key=lambda t: t[1], reverse=True,
        )
        survivors = {tid for tid, _ in ranked[: self.keep_top_k]}
        out: dict[str, TrialDecision] = {}
        for tid, score in ranked:
            alive = tid in survivors
            reason = (
                "top-K by score" if alive
                else f"score {score:.3f} below top-{self.keep_top_k} threshold"
            )
            out[tid] = TrialDecision(
                trial_id=tid, kill=(not alive), reason=reason, score=float(score),
            )
        return out
