"""Robust refusal-direction estimator.

The naive diff-of-means estimator in ``refusal_direction.py`` works on
synthetic activations but is fragile in practice: 8 hand-written prompts
+ mean-pooling-over-tokens + a single hardcoded layer gives a direction
that doesn't actually drive abliteration on real models (verified on
Gemma 4 E2B + Qwen 2.5 3B). This module fixes the three engineering
issues that matter:

1. **Sample size**: 100+ paired prompts per class (use
   ``corpora.HARMFUL_PROMPTS_100`` / ``HARMLESS_PROMPTS_100`` for an
   in-tree corpus, or pass your own).

2. **Pooling strategy**: ``pool="last"`` takes the residual-stream
   activation at the LAST token of the instruction (i.e. just before
   generation begins). This is what Arditi et al. use and what real
   abliterators (Heretic, FailSpy/abliterator) do. Mean-over-tokens
   averages refusal signal with content signal and produces a noisier
   direction.

3. **Layer sweep**: probe several candidate layers, compute a Fisher
   discriminant score for refusal separability per layer, return the
   direction at the most-separable layer. The refusal direction varies
   across depth — picking the wrong layer gives a useless direction.

The output is a ``(RefusalDirection, layer_idx, score)`` tuple. The
single direction `r` is then applied uniformly to every residual-stream-
writing weight matrix during abliteration (standard recipe).
"""

from __future__ import annotations

from typing import Iterable

from kairos.aletheia.surgery.refusal_direction import (
    RefusalDirection, compute_direction_from_activations,
)


def _probe_layer(model, tokenizer, prompts: list[str], layer_idx: int,
                  *, pool: str = "last", device: str = "cuda",
                  apply_chat_template: bool = True):
    """Run a forward pass per prompt; pool the residual-stream hidden
    state at ``layer_idx``.

    Returns a (N, d) tensor on CPU (fp32).
    """
    import torch

    model.eval()
    out: list[object] = []
    with torch.no_grad():
        for p in prompts:
            if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    prompt_text = p
            else:
                prompt_text = p
            enc = tokenizer(prompt_text, return_tensors="pt",
                              truncation=True, max_length=512,
                              add_special_tokens=False).to(device)
            out_dict = model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"],
                               output_hidden_states=True)
            hs = out_dict.hidden_states
            if not (0 <= layer_idx < len(hs)):
                raise IndexError(
                    f"layer_idx={layer_idx} out of range "
                    f"(model has {len(hs)} hidden states)"
                )
            h_layer = hs[layer_idx]                  # (1, T, d)
            if pool == "last":
                h = h_layer[0, -1, :]
            elif pool == "mean":
                h = h_layer[0].mean(dim=0)
            else:
                raise ValueError(f"pool must be 'last' or 'mean'; got {pool!r}")
            out.append(h.detach().float().cpu())
    import torch
    return torch.stack(out, dim=0)                   # (N, d)


def _fisher_score(h_harmful, h_harmless, direction) -> float:
    """How well does `direction` linearly separate the two clusters?

    Score = (mean diff)^2 / (pooled variance) along the direction.
    Higher = better separation.
    """
    import torch

    proj_h = h_harmful.float() @ direction
    proj_b = h_harmless.float() @ direction
    mean_diff = float((proj_h.mean() - proj_b.mean()).item())
    var_pooled = float(((proj_h.var() + proj_b.var()) / 2.0).item())
    return (mean_diff * mean_diff) / max(var_pooled, 1e-6)


def compute_refusal_direction_robust(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    *,
    candidate_layers: Iterable[int] | None = None,
    pool: str = "last",
    device: str = "cuda",
    apply_chat_template: bool = True,
    verbose: bool = True,
) -> tuple[RefusalDirection, int, float]:
    """Find the refusal direction at the most-separable residual-stream
    layer.

    Parameters
    ----------
    model, tokenizer : HF model + tokenizer
    harmful_prompts, harmless_prompts : list[str]
        Paired prompt corpora. Recommend 50+ each; use
        ``kairos.aletheia.surgery.corpora.HARMFUL_PROMPTS_100`` and
        ``HARMLESS_PROMPTS_100`` for sensible defaults.
    candidate_layers : iterable of int | None
        Layer indices to sweep over. ``None`` (default) auto-selects
        every ~10% of depth.
    pool : "last" | "mean"
        Token-pooling strategy. ``"last"`` (default) is canonical.
    device : str
        Where to run forward passes.
    apply_chat_template : bool
        If True (default), wrap each prompt in the model's chat
        template before tokenising. Required for instruction-tuned
        models — without it, Gemma/Qwen/Llama-chat immediately emit
        EOS.
    verbose : bool
        Print per-layer Fisher scores.

    Returns
    -------
    (direction, best_layer, best_score)
        ``direction`` is a ``RefusalDirection`` (unit norm).
        ``best_layer`` is the chosen layer index (0 = embedding, 1+ =
        transformer block outputs).
        ``best_score`` is the Fisher discriminant score at that layer.
    """
    if len(harmful_prompts) != len(harmless_prompts):
        raise ValueError(
            f"corpora must be paired (same length): "
            f"harmful={len(harmful_prompts)} harmless={len(harmless_prompts)}"
        )
    if pool not in ("last", "mean"):
        raise ValueError(f"pool must be 'last' or 'mean'; got {pool!r}")

    # Auto-select candidate layers: probe ~6 evenly-spaced points
    # across depth. We get the depth by running a single forward.
    import torch
    model.eval()
    with torch.no_grad():
        probe = tokenizer("hi", return_tensors="pt",
                            add_special_tokens=False).to(device)
        out = model(**probe, output_hidden_states=True)
        n_hidden = len(out.hidden_states)            # = n_layers + 1

    if candidate_layers is None:
        # Skip layer 0 (embedding) and the very last layer; sample
        # ~6 layers across the middle 80% of depth.
        lo = max(1, int(0.15 * n_hidden))
        hi = max(lo + 1, int(0.95 * n_hidden))
        step = max(1, (hi - lo) // 6)
        candidate_layers = list(range(lo, hi, step))
    else:
        candidate_layers = list(candidate_layers)

    if verbose:
        print(f"  layer sweep: candidates={list(candidate_layers)} "
              f"(model has {n_hidden} hidden states)", flush=True)

    best_layer = -1
    best_score = -1.0
    best_direction: RefusalDirection | None = None
    per_layer = []

    for layer_idx in candidate_layers:
        h_harm = _probe_layer(model, tokenizer, harmful_prompts, layer_idx,
                                pool=pool, device=device,
                                apply_chat_template=apply_chat_template)
        h_safe = _probe_layer(model, tokenizer, harmless_prompts, layer_idx,
                                pool=pool, device=device,
                                apply_chat_template=apply_chat_template)
        r = compute_direction_from_activations(h_harm, h_safe)
        score = _fisher_score(h_harm, h_safe, r.direction)
        per_layer.append((layer_idx, score))
        if verbose:
            print(f"    layer {layer_idx:>3d}: Fisher score = {score:.4f}",
                  flush=True)
        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_direction = r

    assert best_direction is not None
    if verbose:
        print(f"  best layer = {best_layer}  Fisher = {best_score:.4f}",
              flush=True)
    return best_direction, best_layer, best_score
