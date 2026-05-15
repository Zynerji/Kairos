"""Reserved for future architecture growth (beyond base 42B).

Aletheia CORE is pure LoRA fine-tuning that merges back to the base
parameter count. Modules in this package are STUBS, not wired into
the default training loop. Enable explicitly via configs/growth.yaml
when ready to expand.

Growth options (all disabled by default):
- ConfidenceHead        ~1-5M params   (explicit P(correct) output)
- PoolSideFFN           ~100-500M / pool (gated residual, zero-gate init)
- add_experts_to_moe    ~250M / new expert (router surgery required)

Total growth budget recommended: <5% of base (~2B on a 42B base).
"""
from kairos.aletheia.growth.confidence_head import ConfidenceHead
from kairos.aletheia.growth.pool_side_ffn import PoolSideFFN
from kairos.aletheia.growth.expert_addition import add_experts_to_moe

__all__ = ["ConfidenceHead", "PoolSideFFN", "add_experts_to_moe"]
