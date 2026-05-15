"""Per-pool gated side-FFN stub.

GROWTH MODULE — not active in core fine-tuning. Enable via
configs/growth.yaml -> pool_side_ffn.enabled = true.

Adds a gated residual computation alongside the base FFN:
    h_out = h_base + sigmoid(gate) * down(SiLU(up(h_base)))

gate is a scalar Parameter initialized to gate_init (default 0.0).
At init, sigmoid(0) = 0.5 so the side path contributes. For a
true zero-op init set gate_init to a large negative number
(e.g. -10 -> sigmoid ~= 4.5e-5).

Param cost per pool:
    gate_up: H x I  (e.g. 4096 x 4096 = ~17M)
    down:    I x H  (~17M)
    gate:    1 scalar
    Total:   ~2 * H * I per pool

With 9 pools x 4096x4096 -> ~300M added. Scale I down for budget.
"""
from __future__ import annotations


class PoolSideFFN:
    """Gated SwiGLU-style side path; zero-gate init keeps base behavior."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_init: float = -4.0,       # sigmoid(-4) ~= 0.018 ~= near-off
    ):
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("sizes must be positive")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_init = gate_init

    def to_module(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        H, I, g0 = self.hidden_size, self.intermediate_size, self.gate_init

        class _PoolSideFFNModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Linear(H, I, bias=False)
                self.down = nn.Linear(I, H, bias=False)
                self.gate = nn.Parameter(torch.tensor(float(g0)))

            def forward(self, h):
                side = self.down(F.silu(self.up(h)))
                return h + torch.sigmoid(self.gate) * side

        return _PoolSideFFNModule()

    def param_count(self) -> int:
        return 2 * self.hidden_size * self.intermediate_size + 1
