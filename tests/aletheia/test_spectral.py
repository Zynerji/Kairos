import math
import pytest
from kairos.aletheia.torsion.spectral_amp import adaptive_alpha, spectral_weights


def test_healthy_head_alpha_zero():
    # head_std ~= target_std -> alpha ~= 0
    a = adaptive_alpha(head_std=1.0, target_std=1.0)
    assert abs(a) < 1e-9


def test_collapsed_head_alpha_max():
    # head_std -> 0 -> alpha = alpha_max
    a = adaptive_alpha(head_std=1e-20, target_std=1.0, alpha_max=5.0)
    assert a == 5.0


def test_degenerate_target_alpha_zero():
    # target_std = 0 -> not meaningful -> alpha = 0
    a = adaptive_alpha(head_std=1.0, target_std=0.0)
    assert a == 0.0


def test_alpha_monotonic_in_collapse():
    # Smaller head_std (more collapsed) -> larger alpha
    a1 = adaptive_alpha(head_std=0.5, target_std=1.0)
    a2 = adaptive_alpha(head_std=0.1, target_std=1.0)
    a3 = adaptive_alpha(head_std=0.01, target_std=1.0)
    assert a1 < a2 < a3


def test_alpha_clamped_to_max():
    a = adaptive_alpha(head_std=1e-50, target_std=1.0, alpha_max=3.0)
    assert a <= 3.0


def test_alpha_non_negative_when_head_larger():
    # head_std > target_std would give negative log -> clamp to 0
    a = adaptive_alpha(head_std=2.0, target_std=1.0)
    assert a == 0.0


# torch-dependent tests
torch = pytest.importorskip("torch")


def test_spectral_weights_alpha_zero_is_identity():
    t = torch.randn(16)
    w = spectral_weights(t, alpha=0.0)
    assert torch.allclose(w, torch.ones_like(w))


def test_spectral_weights_boosts_extremes():
    t = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0])  # one extreme sample
    w = spectral_weights(t, alpha=2.0)
    # Extreme sample should have highest weight
    assert w.argmax().item() == 4
    assert w[4] > w[0]


def test_spectral_weights_mean_normalized():
    t = torch.randn(32)
    w = spectral_weights(t, alpha=1.5)
    # Mean-normalized: total loss scale preserved
    assert abs(w.mean().item() - 1.0) < 1e-5


def test_spectral_weights_requires_tensor():
    with pytest.raises(TypeError):
        spectral_weights([1.0, 2.0, 3.0], alpha=1.0)


def test_spectral_weights_handles_vector_targets():
    t = torch.randn(8, 4)  # [B, D]
    w = spectral_weights(t, alpha=1.0)
    assert w.shape == (8,)
