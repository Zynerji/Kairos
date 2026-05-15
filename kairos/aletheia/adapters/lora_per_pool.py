"""Per-pool LoRA adapter management via PEFT.

Core fine-tune path: one named LoRA adapter per pool, activated
solo in Phase A. In Phase B all adapters are frozen and the backbone
is selectively unfrozen (router + shared-experts + attention for MoE;
entire backbone for dense).

Adapter names follow Pool.adapter_name: 'aletheia_<pool_name>'.
Merge-to-base produces a 42B checkpoint with no runtime overhead.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoRAPoolConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def _adapter_name(pool_name: str) -> str:
    return pool_name if pool_name.startswith("aletheia_") else f"aletheia_{pool_name}"


def register_pool_adapters(model: Any, pool_names: list[str], cfg: LoRAPoolConfig) -> Any:
    """Attach one LoRA adapter per pool. Returns peft-wrapped model."""
    from peft import LoraConfig, get_peft_model

    if not pool_names:
        raise ValueError("pool_names cannot be empty")

    lora = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias, task_type=cfg.task_type,
    )
    first = _adapter_name(pool_names[0])
    peft_model = get_peft_model(model, lora, adapter_name=first)
    for name in pool_names[1:]:
        peft_model.add_adapter(_adapter_name(name), lora)
    return peft_model


def activate_pool(model: Any, adapter_name: str) -> None:
    """adapter_name: full registered name (e.g. 'aletheia_factuality')."""
    model.set_adapter(adapter_name)


def freeze_all_adapters(model: Any) -> None:
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = False


def unfreeze_all_adapters(model: Any) -> None:
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True


def freeze_backbone(model: Any) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False


def unfreeze_backbone(
    model: Any,
    include_router: bool = True,
    include_shared: bool = True,
    keep_routed_experts_frozen: bool = True,
) -> None:
    """Phase B backbone unfreeze, MoE-aware.

    Routed experts stay frozen (too few tokens per batch for clean
    gradient); spectral amp revives dead experts instead. For dense
    models, pass keep_routed_experts_frozen=False and set defaults.
    """
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        if keep_routed_experts_frozen and "experts." in n and "shared" not in n:
            p.requires_grad = False
            continue
        if "router" in n or ("gate" in n and "gate_proj" not in n):
            p.requires_grad = bool(include_router)
            continue
        if "shared_expert" in n or "shared_experts" in n:
            p.requires_grad = bool(include_shared)
            continue
        p.requires_grad = True


def merge_all_adapters(model: Any, weights: dict[str, float] | None = None) -> Any:
    """Average-merge all adapters into base weights; return unified model.

    Equal weights by default. For per-pool emphasis (e.g. boost factuality),
    supply a dict: {'aletheia_factuality': 1.5, 'aletheia_sycophancy': 1.2, ...}
    """
    if not hasattr(model, "peft_config"):
        raise RuntimeError("model has no peft_config -- register adapters first")

    adapters = list(model.peft_config.keys())
    if weights is None:
        weights = {a: 1.0 / len(adapters) for a in adapters}

    model.add_weighted_adapter(
        adapters=list(weights.keys()),
        weights=list(weights.values()),
        adapter_name="aletheia_merged",
    )
    model.set_adapter("aletheia_merged")
    return model.merge_and_unload()
