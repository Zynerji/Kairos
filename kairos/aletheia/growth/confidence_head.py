"""Confidence head stub — explicit P(correct) output.

GROWTH MODULE — not active in core fine-tuning. Enable via
configs/growth.yaml -> confidence_head.enabled = true.

Adds a small Linear projection on the final hidden state producing
a scalar in [0, 1]. Trained by the calibration pool against
correctness labels using Brier + ECE loss.

Param cost: hidden_size floats (e.g. 4096 for Qwen3-8B -> 4K params,
or hidden x 1 projection ~= 1-5M depending on architecture).

Directly ports jDHART's calibration head (P(profitable) binary).
"""
from __future__ import annotations


class ConfidenceHead:
    """Sigmoid projection from last hidden state.

    Lazy torch import so the scaffold remains importable without torch.
    Instantiation requires torch; call .to_module() to get a real nn.Module.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.dropout = dropout

    def to_module(self):
        import torch
        import torch.nn as nn

        class _ConfidenceHeadModule(nn.Module):
            def __init__(self, hidden_size: int, dropout: float):
                super().__init__()
                self.drop = nn.Dropout(dropout)
                self.proj = nn.Linear(hidden_size, 1)

            def forward(self, hidden_states):
                # [B, T, H] -> take last token's hidden state
                h = hidden_states[:, -1, :]
                h = self.drop(h)
                logit = self.proj(h).squeeze(-1)
                return torch.sigmoid(logit)

        return _ConfidenceHeadModule(self.hidden_size, self.dropout)

    def param_count(self) -> int:
        return self.hidden_size + 1  # Linear(H, 1) = H weights + 1 bias
