"""Path A — capability-aware abliteration: do the abliteration ourselves
but orthogonalise the refusal direction against a capability subspace
before projection, so the projection only removes the
refusal-orthogonal-to-capability content.

Output is a state-dict that's abliterated against refusal but preserves
capability-correlated content the standard recipe (Arditi et al.) would
have damaged.

Steps:

  1. Load the un-abliterated base model (any open-weight HF causal-LM).
  2. Probe residual-stream activations on harmful / harmless /
     per-capability-axis prompts. Compute refusal direction r and
     capability subspace C.
  3. Build a CapabilityAwareAbliterator. .prepare() reports per-axis
     overlap (how much of the refusal direction lies along each axis —
     i.e. which capabilities the standard recipe would have damaged).
  4. .apply() to the state-dict. Save the modified weights and the
     codebook (r_pure, C, overlaps, touched layers).

Run on the VM (needs the un-abliterated base, in this example
google/gemma-4-E2B-it which is Apache 2.0 / public)::

    PYTHONPATH=. python3 examples/aletheia_capability_aware_abliterate.py \\
        --model google/gemma-4-E2B-it \\
        --save-dir ./capability_aware_abliterated
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos.aletheia.surgery import (  # noqa: E402
    CapabilityAwareAbliterator,
    compute_capability_subspace,
    compute_direction_from_activations,
)

# Re-use the prompt sets from the codebook script.
from examples.aletheia_codebook_restore import (  # noqa: E402
    HARMFUL_PROMPTS, HARMLESS_PROMPTS, CAPABILITY_PROMPTS,
    NEUTRAL_PROMPTS, probe_activations,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                          default="google/gemma-4-E2B-it",
                          help="un-abliterated base model to abliterate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--save-dir", type=str,
                          default="./capability_aware_abliterated")
    parser.add_argument("--target-suffixes", type=str,
                          default="o_proj.weight,down_proj.weight,o_proj.linear.weight,down_proj.linear.weight",
                          help="comma-separated weight-key suffixes to project")
    parser.add_argument("--skip-substrings", type=str,
                          default="vision_tower,audio_tower,multi_modal_projector,embed_audio,embed_vision",
                          help="comma-separated substrings — skip any key containing these")
    parser.add_argument("--save-safetensors", action="store_true",
                          help="save the modified weights as safetensors "
                          "(default: only save the codebook)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Probes
    print("\nprobing harmful / harmless / capability axes...", flush=True)
    h_harm = probe_activations(model, tok, HARMFUL_PROMPTS,
                                 layer_idx=args.layer_idx, device=args.device)
    h_safe = probe_activations(model, tok, HARMLESS_PROMPTS,
                                 layer_idx=args.layer_idx, device=args.device)
    h_neut = probe_activations(model, tok, NEUTRAL_PROMPTS,
                                 layer_idx=args.layer_idx, device=args.device)
    axis_acts = {
        name: probe_activations(model, tok, prompts,
                                 layer_idx=args.layer_idx, device=args.device)
        for name, prompts in CAPABILITY_PROMPTS.items()
    }

    refusal = compute_direction_from_activations(h_harm, h_safe)
    capability = compute_capability_subspace(axis_acts, h_neut)

    print(f"\nrefusal direction norm: {float(refusal.direction.norm()):.6f}",
          flush=True)
    print(f"capability subspace rank: {capability.basis.shape[1]} "
          f"axes={capability.axis_names}", flush=True)

    # Apply
    target_suffixes = [s.strip() for s in args.target_suffixes.split(",") if s.strip()]
    skip_substrings = [s.strip() for s in args.skip_substrings.split(",") if s.strip()]
    abl = CapabilityAwareAbliterator(
        refusal, capability,
        target_suffixes=target_suffixes,
        skip_substrings=skip_substrings,
    )
    rep_prep = abl.prepare()
    print(f"\nrefusal norm before orthogonalise: {rep_prep.refusal_norm_before:.6f}",
          flush=True)
    print(f"refusal norm after  orthogonalise: {rep_prep.refusal_norm_after_orthogonalise:.6f}",
          flush=True)
    print(f"per-axis overlap (standard abliteration would have damaged these):")
    for axis, overlap in rep_prep.axis_overlaps.items():
        print(f"  {axis:>16s}  {overlap:+.4f}", flush=True)

    # State dict on CPU for the surgery
    print("\napplying capability-aware abliteration to state_dict...",
          flush=True)
    cpu_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    new_sd = abl.apply(cpu_sd)
    print(f"  touched={abl.report.n_touched}  "
          f"skipped={abl.report.n_skipped}", flush=True)

    # Save codebook
    codebook = abl.export_codebook()
    # Pure tensors → torch.save
    torch.save({
        "r_pure": codebook["r_pure"],
        "capability_basis": codebook["capability_basis"],
        "capability_axes": codebook["capability_axes"],
        "axis_overlaps": codebook["axis_overlaps"],
        "touched_layers": codebook["touched_layers"],
        "source_model": args.model,
    }, save_dir / "codebook.pt")
    print(f"\nwrote {save_dir / 'codebook.pt'}", flush=True)
    (save_dir / "report.json").write_text(json.dumps({
        "source_model": args.model,
        "n_touched": abl.report.n_touched,
        "n_skipped": abl.report.n_skipped,
        "refusal_norm_before": rep_prep.refusal_norm_before,
        "refusal_norm_after_orthogonalise": rep_prep.refusal_norm_after_orthogonalise,
        "axis_overlaps": rep_prep.axis_overlaps,
        "touched_layers": rep_prep.touched_layers,
    }, indent=2))
    print(f"wrote {save_dir / 'report.json'}", flush=True)

    if args.save_safetensors:
        try:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in new_sd.items()},
                      str(save_dir / "model.safetensors"))
            print(f"wrote {save_dir / 'model.safetensors'}", flush=True)
        except ImportError:
            print("safetensors not available; skipping weight save", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
