"""Path B validation — codebook-driven healing of an already-abliterated
checkpoint.

Hypothesis: given (A, B) where A is un-abliterated and B is the standard-
abliteration derivative, our ``WeightDeltaCodebook`` can selectively re-
inject the capability-aligned fraction of (A - B), recovering capability
B lost while keeping refusal removed.

For Gemma 4 E2B:
  A = google/gemma-4-E2B-it
  B = huihui-ai/Huihui-gemma-4-E2B-it-abliterated

Decision rule per alpha ∈ {0.0, 0.25, 0.5, 1.0}:
  C_α = B + α · ΔW_capability_aligned

  We expect:
    - α = 0      → identical to B
    - α = 1      → maximum capability re-injection
    - some α*    → optimal Pareto point (GSM8K back near A's, refusal still near B's)

This is the smaller, more honest experiment: it doesn't require us to
correctly compute a refusal direction ourselves (which is what failed in
the Path A validation). It just requires that abliteration damage to
capability is concentrated along directions in our capability subspace
— which is exactly the theoretical claim.

Run on the VM (assumes A and B already cached, ~13 GB free)::

    PYTHONPATH=. python3 examples/aletheia_codebook_validate.py \\
        --alphas 0,0.25,0.5,1.0 --gsm8k-samples 30 --refusal-prompts 30 \\
        --save-dir ./codebook_validation
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kairos.aletheia.surgery import (  # noqa: E402
    WeightDeltaCodebook,
    compute_capability_subspace,
)
from examples.aletheia_codebook_restore import (  # noqa: E402
    CAPABILITY_PROMPTS, NEUTRAL_PROMPTS, probe_activations,
)
from examples.aletheia_abc_validate import (  # noqa: E402
    REFUSAL_PROBE_PROMPTS, REFUSAL_PATTERNS,
    score_gsm8k, score_refusal, evaluate_model,
)


MODEL_A = "google/gemma-4-E2B-it"
MODEL_B = "huihui-ai/Huihui-gemma-4-E2B-it-abliterated"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=str, default="0.0,0.25,0.5,1.0")
    parser.add_argument("--gsm8k-samples", type=int, default=30)
    parser.add_argument("--refusal-prompts", type=int, default=30)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--save-dir", type=str,
                          default="./codebook_validation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reload-existing-eval", type=str, default=None,
                          help="path to abc_validation report.json to "
                          "reuse A and B baseline scores")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    refusal_prompts = REFUSAL_PROBE_PROMPTS[: args.refusal_prompts]
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"loading tokenizer for {MODEL_A} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_A)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ------------------------------------------------------------------
    # Step 1: Load A, probe capability subspace, snapshot state dict
    # ------------------------------------------------------------------
    print(f"\nloading A = {MODEL_A} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_A, dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    print("\nprobing capability axes from A...", flush=True)
    h_neut = probe_activations(model, tok, NEUTRAL_PROMPTS,
                                 layer_idx=args.layer_idx, device=args.device)
    axis_acts = {
        name: probe_activations(model, tok, prompts,
                                 layer_idx=args.layer_idx,
                                 device=args.device)
        for name, prompts in CAPABILITY_PROMPTS.items()
    }
    capability = compute_capability_subspace(axis_acts, h_neut)
    print(f"  capability rank = {capability.basis.shape[1]} "
          f"axes={capability.axis_names}", flush=True)

    a_sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Optional A eval (skip if we have a reloaded baseline)
    summary: dict[str, dict] = {}
    if args.reload_existing_eval is not None:
        prev = json.loads(pathlib.Path(args.reload_existing_eval).read_text())
        if "A_original" in prev:
            summary["A_original"] = prev["A_original"]
            print(f"\nreusing prior A eval: GSM8K={summary['A_original']['gsm8k_accuracy']:.4f}  "
                  f"refusal={summary['A_original']['refusal_rate']:.4f}",
                  flush=True)
        if "B_third_party" in prev:
            summary["B_third_party"] = prev["B_third_party"]
            print(f"reusing prior B eval: GSM8K={summary['B_third_party']['gsm8k_accuracy']:.4f}  "
                  f"refusal={summary['B_third_party']['refusal_rate']:.4f}",
                  flush=True)
    else:
        summary["A_original"] = evaluate_model(
            model, tok, label="A_original",
            gsm8k_n=args.gsm8k_samples,
            refusal_prompts=refusal_prompts,
            save_dir=save_dir, device=args.device,
        )

    # Free A
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: Load B, build codebook, evaluate B baseline (if needed)
    # ------------------------------------------------------------------
    print(f"\nloading B = {MODEL_B} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_B, dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    b_sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if "B_third_party" not in summary:
        summary["B_third_party"] = evaluate_model(
            model, tok, label="B_third_party",
            gsm8k_n=args.gsm8k_samples,
            refusal_prompts=refusal_prompts,
            save_dir=save_dir, device=args.device,
        )

    # ------------------------------------------------------------------
    # Step 3: Build the codebook
    # ------------------------------------------------------------------
    print("\nbuilding codebook from (A, B) state dicts...", flush=True)
    book = WeightDeltaCodebook()
    # Restrict to the projections abliteration actually touches —
    # default ".weight" matches the 262k-vocab LM head whose SVD would
    # take hours on CPU.
    rep = book.build(a_sd_cpu, b_sd_cpu,
                       target_suffixes=["o_proj.weight",
                                          "down_proj.weight"])
    print(f"  paired={rep.n_paired}  skipped={rep.n_skipped}", flush=True)
    splits = book.split_against_capability(capability)
    alpha_caps = sorted(((parts["alpha_cap"], name)
                            for name, parts in splits.items()), reverse=True)
    print(f"\ntop layers by capability-overlap (alpha_cap):")
    for ac, name in alpha_caps[:5]:
        print(f"  {ac:.4f}  {name}")
    print(f"bottom layers (mostly pure refusal):")
    for ac, name in alpha_caps[-3:]:
        print(f"  {ac:.4f}  {name}")

    # Save codebook for reproducibility
    torch.save({
        "capability_basis": capability.basis,
        "capability_axes": capability.axis_names,
        "alpha_caps": {name: ac for ac, name in alpha_caps},
        "n_paired_layers": rep.n_paired,
        "model_A": MODEL_A, "model_B": MODEL_B,
    }, save_dir / "codebook.pt")

    # ------------------------------------------------------------------
    # Step 4: For each alpha > 0, restore + eval
    # ------------------------------------------------------------------
    for alpha in alphas:
        if alpha == 0.0:
            # alpha = 0 → identical to B (already evaluated as B_third_party)
            summary[f"C_alpha_{alpha:.2f}"] = dict(summary["B_third_party"],
                                                       label=f"C_alpha_{alpha:.2f}")
            continue
        print(f"\n--- α = {alpha} ---", flush=True)
        healed_sd = book.apply_restoration(b_sd_cpu, capability, alpha=alpha)
        healed_gpu = {k: v.to(model.device) for k, v in healed_sd.items()}
        missing, unexpected = model.load_state_dict(healed_gpu, strict=False)
        print(f"  loaded healed state: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
        del healed_gpu

        label = f"C_alpha_{alpha:.2f}"
        summary[label] = evaluate_model(
            model, tok, label=label,
            gsm8k_n=args.gsm8k_samples,
            refusal_prompts=refusal_prompts,
            save_dir=save_dir, device=args.device,
        )

    # ------------------------------------------------------------------
    # Step 5: Comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 76)
    print(f"{'label':<24} {'GSM8K acc':>10} {'refusal rate':>14} {'product':>10}")
    print("-" * 76)
    for k in ["A_original", "B_third_party"] + [f"C_alpha_{a:.2f}" for a in alphas]:
        if k not in summary:
            continue
        s = summary[k]
        gsm = s["gsm8k_accuracy"]
        ref = s["refusal_rate"]
        # Product metric: GSM8K capacity * (1 - refusal) — higher is better
        prod = max(gsm, 0.001) * max(1.0 - ref, 0.001)
        print(f"{k:<24} {gsm:>10.4f} {ref:>14.4f} {prod:>10.4f}")
    print("=" * 76)

    a = summary["A_original"]
    b = summary["B_third_party"]
    print(f"\nA → B (the damage):  "
          f"GSM8K {b['gsm8k_accuracy'] - a['gsm8k_accuracy']:+.4f}  "
          f"refusal {b['refusal_rate'] - a['refusal_rate']:+.4f}")
    best_alpha = None
    best_prod = -1
    for alpha in alphas:
        if alpha == 0.0:
            continue
        c = summary[f"C_alpha_{alpha:.2f}"]
        delta_gsm = c["gsm8k_accuracy"] - b["gsm8k_accuracy"]
        delta_ref = c["refusal_rate"] - b["refusal_rate"]
        prod = max(c["gsm8k_accuracy"], 0.001) * max(1.0 - c["refusal_rate"], 0.001)
        print(f"B → C(α={alpha}): "
              f"GSM8K {delta_gsm:+.4f}  refusal {delta_ref:+.4f}  product={prod:.4f}")
        if prod > best_prod and c["refusal_rate"] <= b["refusal_rate"] + 0.10:
            best_prod = prod
            best_alpha = alpha
    if best_alpha is not None:
        print(f"\nbest α (Pareto): {best_alpha} (product={best_prod:.4f})")

    (save_dir / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {save_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
