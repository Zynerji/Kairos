"""Torsion training orchestrator.

Runs alternating Phase A (per-pool solo adapter training, backbone frozen)
and Phase B (adapters frozen, backbone unfrozen, distill/ranking loss).

Per jDHART v22 lesson: train each pool solo in Phase A -- no gradient
interference through the shared backbone.
Per jDHART v20 lesson: heads-only saturates at ~cycle 25; torsion required.
Per v20-pareto2: ratchet 80% floor, product metric, dual-regression rollback.

The orchestrator is model-library-agnostic -- all torch-specific operations
are injected as callable hooks by the training script.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable
import time

from kairos.aletheia.torsion.bronze import BronzePendulum
from kairos.aletheia.torsion.torus import TorusPendulum
from kairos.aletheia.ratchet.pareto import ParetoRatchet
from kairos.aletheia.pools.base import Pool


@dataclass
class TorsionState:
    cycle: int = 0
    last_scores: dict[str, float] = field(default_factory=dict)
    last_phase_a_loss: dict[str, float] = field(default_factory=dict)
    last_phase_b_loss: float = 0.0
    rollbacks: int = 0
    new_bests: int = 0


@dataclass
class TorsionCycle:
    pools: list[Pool]
    ratchet: ParetoRatchet
    bronze: BronzePendulum
    torus: TorusPendulum
    phase_b_steps: int = 50
    batch_size: int = 4

    # Model hooks (training script supplies real torch-backed implementations)
    activate_pool_adapter: Callable[[Any, str], None] = None         # (model, adapter_name)
    freeze_backbone: Callable[[Any], None] = None
    unfreeze_backbone: Callable[[Any], None] = None
    freeze_adapters: Callable[[Any], None] = None
    unfreeze_adapters: Callable[[Any], None] = None
    phase_b_step: Callable[[Any, int], float] = None                  # (model, step) -> scalar loss
    save_checkpoint: Callable[[Any, Path], Path] = None               # (model, path) -> path written
    restore_checkpoint: Callable[[Any, Path], None] = None
    optimizer_step: Callable[[Any], None] = None                      # opt.step(); opt.zero_grad()

    state: TorsionState = field(default_factory=TorsionState)

    def __post_init__(self) -> None:
        if not self.pools:
            raise ValueError("TorsionCycle requires at least one pool")
        pool_names = [p.name for p in self.pools]
        if len(set(pool_names)) != len(pool_names):
            raise ValueError(f"pool names must be unique, got {pool_names}")
        # All hooks are required; fail fast rather than cryptically later
        missing = [k for k, v in {
            "activate_pool_adapter": self.activate_pool_adapter,
            "freeze_backbone": self.freeze_backbone,
            "unfreeze_backbone": self.unfreeze_backbone,
            "freeze_adapters": self.freeze_adapters,
            "unfreeze_adapters": self.unfreeze_adapters,
            "phase_b_step": self.phase_b_step,
            "save_checkpoint": self.save_checkpoint,
            "restore_checkpoint": self.restore_checkpoint,
            "optimizer_step": self.optimizer_step,
        }.items() if v is None]
        if missing:
            raise ValueError(f"missing hooks: {missing}")

    def run(
        self,
        model: Any,
        max_cycles: int,
        output_dir: Path,
        log_cb: Callable[[dict], None] | None = None,
    ) -> TorsionState:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for cycle in range(max_cycles):
            self.state.cycle = cycle
            t0 = time.time()

            self._phase_a(model, cycle)
            self._phase_b(model, cycle)

            scores = self._eval_oot(model)
            self.state.last_scores = scores
            event = self._ratchet_step(model, scores, output_dir)

            if log_cb:
                log_cb({
                    "cycle": cycle,
                    "scores": scores,
                    "event": event,
                    "elapsed_s": time.time() - t0,
                    "phase_a_loss": dict(self.state.last_phase_a_loss),
                    "phase_b_loss": self.state.last_phase_b_loss,
                    "rollbacks_total": self.state.rollbacks,
                    "new_bests_total": self.state.new_bests,
                    "best_product": self.ratchet.best_product,
                })

        return self.state

    def _phase_a(self, model: Any, cycle: int) -> None:
        self.freeze_backbone(model)
        self.unfreeze_adapters(model)
        self.state.last_phase_a_loss.clear()

        for pool in self.pools:
            self.activate_pool_adapter(model, pool.adapter_name)
            sub_steps = self.torus.step_count(pool.name, cycle)
            total_loss = 0.0
            count = 0
            for step, batch in enumerate(islice(pool.train_loader(self.batch_size), sub_steps)):
                w = self.bronze.weight(pool.name, step)
                loss = pool.loss(batch, model, scale=w)
                loss.backward()
                self.optimizer_step(model)
                total_loss += float(loss.detach())
                count += 1
            self.state.last_phase_a_loss[pool.name] = total_loss / max(count, 1)

    def _phase_b(self, model: Any, cycle: int) -> None:
        self.freeze_adapters(model)
        self.unfreeze_backbone(model)
        total = 0.0
        for step in range(self.phase_b_steps):
            total += float(self.phase_b_step(model, step))
        self.state.last_phase_b_loss = total / max(self.phase_b_steps, 1)

    def _eval_oot(self, model: Any) -> dict[str, float]:
        return {p.name: p.evaluate(model, batch_size=self.batch_size).score for p in self.pools}

    def _ratchet_step(self, model: Any, scores: dict[str, float], output_dir: Path) -> str:
        if self.ratchet.should_rollback(scores):
            if self.ratchet.best_checkpoint is not None:
                self.restore_checkpoint(model, self.ratchet.best_checkpoint)
                self.state.rollbacks += 1
                return "rollback"
            return "rollback_no_anchor"
        if self.ratchet.is_new_best(scores):
            ckpt = self.save_checkpoint(
                model, output_dir / f"pareto-best-cycle-{self.state.cycle}"
            )
            self.ratchet.update(scores, ckpt)
            self.state.new_bests += 1
            return "new_best"
        return "no_event"
