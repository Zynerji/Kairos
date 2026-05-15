"""Dev training -- Qwen3-8B-abliterated, single GPU.

Debug the torsion loop with cheap iterations before spending on the 42B run.

Usage:
    python scripts/train_dev_8b.py --max-cycles 5
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml


def _load_configs(base_path: str, ratchet_path: str, pendulums_path: str) -> tuple[dict, dict, dict]:
    base = yaml.safe_load(Path(base_path).read_text())
    ratchet = yaml.safe_load(Path(ratchet_path).read_text())
    pend = yaml.safe_load(Path(pendulums_path).read_text())
    return base, ratchet, pend


def _build_pools(tokenizer, cfg):
    from kairos.aletheia.pools.factuality import FactualityPool
    from kairos.aletheia.pools.calibration import CalibrationPool
    from kairos.aletheia.pools.abstention import AbstentionPool
    from kairos.aletheia.pools.grounding import GroundingPool
    from kairos.aletheia.pools.consistency import ConsistencyPool
    from kairos.aletheia.pools.sycophancy import SycophancyPool
    from kairos.aletheia.pools.reasoning import ReasoningPool
    from kairos.aletheia.pools.instruction import InstructionPool
    from kairos.aletheia.pools.distillation import DistillationPool

    return [
        ReasoningPool(tokenizer=tokenizer),
        FactualityPool(tokenizer=tokenizer),
        CalibrationPool(tokenizer=tokenizer),
        AbstentionPool(tokenizer=tokenizer),
        GroundingPool(tokenizer=tokenizer),
        ConsistencyPool(tokenizer=tokenizer),
        SycophancyPool(tokenizer=tokenizer),
        InstructionPool(tokenizer=tokenizer),
        DistillationPool(tokenizer=tokenizer),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",           default="configs/base_dev.yaml")
    ap.add_argument("--ratchet-config",   default="configs/ratchet.yaml")
    ap.add_argument("--pendulums-config", default="configs/pendulums.yaml")
    ap.add_argument("--max-cycles",       type=int, default=10)
    ap.add_argument("--output-dir",       default="./weights/dev-8b")
    args = ap.parse_args()

    cfg, rcfg, pcfg = _load_configs(args.config, args.ratchet_config, args.pendulums_config)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kairos.aletheia.torsion.bronze import BronzePendulum
    from kairos.aletheia.torsion.torus import TorusPendulum
    from kairos.aletheia.torsion.cycle import TorsionCycle
    from kairos.aletheia.ratchet.pareto import ParetoRatchet
    from kairos.aletheia.adapters.lora_per_pool import (
        LoRAPoolConfig, register_pool_adapters, activate_pool,
        freeze_all_adapters, unfreeze_all_adapters,
        freeze_backbone, unfreeze_backbone,
    )

    repo = cfg["model"]["repo_id"]
    dtype = getattr(torch, cfg["model"]["dtype"])
    print(f"[aletheia] loading tokenizer: {repo}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    print(f"[aletheia] loading model: {repo} dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, trust_remote_code=True, device_map="auto",
    )
    if cfg["compute"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    pools = _build_pools(tokenizer, cfg)
    pool_names = [p.name for p in pools]
    print(f"[aletheia] pools: {pool_names}", flush=True)

    lora_cfg = LoRAPoolConfig(
        rank=cfg["adapters"]["rank"],
        alpha=cfg["adapters"]["alpha"],
        dropout=cfg["adapters"]["dropout"],
        target_modules=tuple(cfg["adapters"]["target_modules"]),
    )
    model = register_pool_adapters(model, pool_names, lora_cfg)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["phase_a"]["lr"],
    )

    def _opt_step(_m):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def _save_ckpt(m, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m.save_pretrained(str(path))
        return path

    def _restore_ckpt(m, path):
        m.load_adapter(str(path), adapter_name="aletheia_restored")
        m.set_adapter("aletheia_restored")

    from kairos.aletheia.phase_b import PhaseBLoss
    distill_pool = next(p for p in pools if p.name == "distillation")
    distill_iter = distill_pool.train_loader(cfg["compute"]["batch_size"])
    phase_b_loss = PhaseBLoss(
        distill_loader=distill_iter,
        distill_ce_weight=cfg["training"]["phase_b"]["loss"]["distill_ce_weight"],
        distill_kl_weight=cfg["training"]["phase_b"]["loss"]["distill_kl_weight"],
        brier_weight=cfg["training"]["phase_b"]["loss"]["calibration_brier_weight"],
        ranking_weight=cfg["training"]["phase_b"]["loss"]["ranking_weight"],
    )

    def _phase_b_step(m, step):
        v = phase_b_loss.step(m, step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return v

    bronze = BronzePendulum(
        heads=pool_names,
        amplitude=pcfg["bronze"]["amplitude"],
        floor=pcfg["bronze"]["floor"],
        ceil=pcfg["bronze"]["ceil"],
    )
    torus = TorusPendulum(
        heads=pool_names,
        weight_amplitude=pcfg["torus"]["weight_amplitude"],
        step_amplitude=pcfg["torus"]["step_amplitude"],
        floor=pcfg["torus"]["floor"],
        ceil=pcfg["torus"]["ceil"],
        base_steps=pcfg["torus"]["base_steps"],
    )
    ratchet = ParetoRatchet(
        anchor={k: rcfg["anchor"][k] for k in pool_names if k in rcfg["anchor"]},
        floor=rcfg["floor"],
        eps=rcfg["eps"],
    )

    cycle = TorsionCycle(
        pools=pools, ratchet=ratchet, bronze=bronze, torus=torus,
        phase_b_steps=cfg["training"]["phase_b"]["steps"],
        batch_size=cfg["compute"]["batch_size"],
        activate_pool_adapter=activate_pool,
        freeze_backbone=freeze_backbone,
        unfreeze_backbone=unfreeze_backbone,
        freeze_adapters=freeze_all_adapters,
        unfreeze_adapters=unfreeze_all_adapters,
        phase_b_step=_phase_b_step,
        save_checkpoint=_save_ckpt,
        restore_checkpoint=_restore_ckpt,
        optimizer_step=_opt_step,
    )

    out_dir = Path(args.output_dir)
    log_path = out_dir / "cycle_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a") as fh:
        def log_cb(record):
            print(
                f"[aletheia] cycle {record['cycle']} "
                f"event={record['event']} "
                f"elapsed={record['elapsed_s']:.1f}s "
                f"best_product={record['best_product']:.4f}",
                flush=True,
            )
            fh.write(json.dumps(record) + "\n")
            fh.flush()

        state = cycle.run(model=model, max_cycles=args.max_cycles, output_dir=out_dir, log_cb=log_cb)

    print(
        f"[aletheia] done. cycles={state.cycle + 1} "
        f"rollbacks={state.rollbacks} new_bests={state.new_bests}",
        flush=True,
    )


if __name__ == "__main__":
    main()
