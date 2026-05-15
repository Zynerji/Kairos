"""Spectral amplification — adaptive alpha per head.

Inverse of VolSurf attenuation. Detects variance collapse in a head's
outputs relative to the target distribution and boosts gradient
through extreme-target samples, providing signal even when overall
correlation is near zero.

alpha = -log(head_std / target_std), clamped to [0, alpha_max]

Healthy head: head_std ~= target_std  -> alpha -> 0     -> no-op
Collapsed head: head_std -> 0          -> alpha -> alpha_max -> strong amp

Per-sample weight:
    amp[i] = exp(alpha * z2[i] / z2_max)
where z2[i] is the squared z-score of target[i] in the batch.
"""
from __future__ import annotations
import math
from typing import Any

EPS = 1e-8


def adaptive_alpha(
    head_std: float,
    target_std: float,
    alpha_max: float = 5.0,
    eps: float = EPS,
) -> float:
    """Compute adaptive alpha from head and target standard deviations."""
    if target_std < eps:
        return 0.0
    if head_std < eps:
        return alpha_max
    ratio = head_std / target_std
    alpha = -math.log(ratio)
    return max(0.0, min(alpha_max, alpha))


def spectral_weights(targets: Any, alpha: float, eps: float = EPS) -> Any:
    """Per-sample amplification weights given a batch of targets.

    targets: torch.Tensor of shape [B] or [B, D].
    Returns torch.Tensor of shape [B], mean-normalized (total loss scale preserved).
    """
    import torch

    if not isinstance(targets, torch.Tensor):
        raise TypeError("spectral_weights requires torch.Tensor")
    if alpha <= 0.0:
        return torch.ones(targets.shape[0], device=targets.device, dtype=targets.dtype)

    t = targets if targets.ndim == 1 else targets.norm(dim=-1)
    mu = t.mean()
    std = t.std().clamp_min(eps)
    z2 = ((t - mu) / std) ** 2
    z2_max = z2.max().clamp_min(eps)
    amp = torch.exp(alpha * z2 / z2_max)
    return amp / amp.mean().clamp_min(eps)
