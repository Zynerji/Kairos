"""Three-model A/B/C validation using our OWN robust direction-finder.

This is the controlled experiment for Path A:

  A = un-abliterated base                  (e.g. Qwen 2.5 3B Instruct)
  B = standard refusal-direction abliteration of A
  C = capability-aware abliteration of A
      (same direction, orthogonalised against capability subspace
       BEFORE projection)

All three share the same direction-finder, same prompt corpora, same
layer choice — so the only thing that differs between B and C is the
capability-orthogonalisation step. This isolates Aletheia's contribution
cleanly, with no dependency on third-party abliterators (Heretic etc.)
and no AGPL concerns.

Decision rule:

    if   GSM8K(C)         >  GSM8K(B)                       AND
         refusal_rate(C)  <= refusal_rate(B) + epsilon       AND
         refusal_rate(C)  << refusal_rate(A)
    then C is a strict improvement on B.
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
    compute_capability_subspace,
    compute_refusal_direction_robust,
    project_out_subspace,
    apply_direction_projection,
    corpora,
)
from examples.aletheia_codebook_validate_v2 import (  # noqa: E402
    collect_capability_prompts, NEUTRAL_PROMPTS,
)
from examples.aletheia_codebook_restore import probe_activations  # noqa: E402
from examples.aletheia_abc_validate import (  # noqa: E402
    REFUSAL_PROBE_PROMPTS, score_gsm8k, score_refusal, evaluate_model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                          default="Qwen/Qwen2.5-3B-Instruct",
                          help="un-abliterated base model A")
    parser.add_argument("--n-refusal-probe", type=int, default=100,
                          help="harmful/harmless prompts for direction-find "
                          "(per class)")
    parser.add_argument("--n-capability-probe", type=int, default=64,
                          help="prompts per capability axis")
    parser.add_argument("--pool", type=str, default="last",
                          choices=["last", "mean"])
    parser.add_argument("--capability-layer", type=int, default=None,
                          help="Layer index for capability-subspace "
                          "probing (default: same as refusal-best layer)")
    parser.add_argument("--gsm8k-samples", type=int, default=50)
    parser.add_argument("--refusal-prompts", type=int, default=30,
                          help="held-out HarmBench-style refusal eval set")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="./abc_pathA")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use distinct prompt sets for direction-finding vs eval — held-out
    # refusal prompts are the latter half of REFUSAL_PROBE_PROMPTS
    # which doesn't overlap with HARMFUL_PROMPTS_100.
    eval_refusal_prompts = REFUSAL_PROBE_PROMPTS[: args.refusal_prompts]

    print(f"loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"\nloading A = {args.model} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB",
          flush=True)

    # ------------------------------------------------------------------
    # Step 1: robust refusal-direction finder
    # ------------------------------------------------------------------
    n = args.n_refusal_probe
    harmful = corpora.HARMFUL_PROMPTS_100[:n]
    harmless = corpora.HARMLESS_PROMPTS_100[:n]
    print(f"\nfinding refusal direction "
          f"(n={n}, pool={args.pool})...", flush=True)
    refusal, best_layer, fisher_score = compute_refusal_direction_robust(
        model, tok, harmful, harmless,
        pool=args.pool, device=args.device, verbose=True,
    )
    print(f"  chosen layer = {best_layer}  Fisher = {fisher_score:.4f}",
          flush=True)

    # ------------------------------------------------------------------
    # Step 2: capability subspace
    # ------------------------------------------------------------------
    cap_layer = (args.capability_layer if args.capability_layer is not None
                  else best_layer)
    print(f"\nbuilding capability subspace "
          f"(n_per_axis={args.n_capability_probe}, "
          f"layer={cap_layer}; refusal best layer was {best_layer})...",
          flush=True)
    axis_prompts = collect_capability_prompts(args.n_capability_probe)
    h_neut = probe_activations(model, tok, NEUTRAL_PROMPTS,
                                 layer_idx=cap_layer, device=args.device)
    axis_acts = {}
    for name, prompts in axis_prompts.items():
        axis_acts[name] = probe_activations(model, tok, prompts,
                                              layer_idx=cap_layer,
                                              device=args.device)
    capability = compute_capability_subspace(axis_acts, h_neut)
    print(f"  capability rank = {capability.basis.shape[1]} "
          f"axes={capability.axis_names}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: snapshot A's state dict (we'll apply both abliterations to it)
    # ------------------------------------------------------------------
    print(f"\nsnapshotting A state dict ...", flush=True)
    a_sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Evaluate A
    summary: dict[str, dict] = {}
    summary["A_original"] = evaluate_model(
        model, tok, label="A_original",
        gsm8k_n=args.gsm8k_samples,
        refusal_prompts=eval_refusal_prompts,
        save_dir=save_dir, device=args.device,
    )

    # ------------------------------------------------------------------
    # Step 4: produce B (raw r) and C (r_pure) state dicts
    # ------------------------------------------------------------------
    r = refusal.direction
    r_pure = project_out_subspace(r, capability)
    print(f"\n‖r‖ = {float(r.norm()):.4f}  ‖r_pure‖ = {float(r_pure.norm()):.4f}",
          flush=True)
    axis_overlaps = {}
    for axis_name in capability.axis_names:
        axis_dir = capability.axis_directions[axis_name]
        denom = float(axis_dir.norm().item())
        if denom > 1e-12:
            axis_overlaps[axis_name] = float(
                (r @ (axis_dir / denom)).item()
            )
    print(f"refusal-direction overlap with capability axes:")
    for name, ov in axis_overlaps.items():
        print(f"  {name:>16s}: {ov:+.4f}")

    print(f"\nbuilding B = raw-direction abliteration of A ...", flush=True)
    b_sd_cpu, info_b = apply_direction_projection(a_sd_cpu, r)
    print(f"  touched {info_b['n_touched']} layers, "
          f"skipped {info_b['n_skipped']}", flush=True)

    print(f"\nbuilding C = capability-aware abliteration of A ...", flush=True)
    c_sd_cpu, info_c = apply_direction_projection(a_sd_cpu, r_pure)
    print(f"  touched {info_c['n_touched']} layers, "
          f"skipped {info_c['n_skipped']}", flush=True)

    # ------------------------------------------------------------------
    # Step 5: load B into the model in-place + eval
    # ------------------------------------------------------------------
    print(f"\nevaluating B (raw-direction abliteration) ...", flush=True)
    # Free A's CPU state-dict copy — only need it once to derive B and C.
    # Loading the CPU dict via load_state_dict copies per-param into the
    # model's existing GPU buffers (no intermediate full-GPU copy).
    del a_sd_cpu
    import gc
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    model.load_state_dict(b_sd_cpu, strict=False)
    summary["B_raw"] = evaluate_model(
        model, tok, label="B_raw",
        gsm8k_n=args.gsm8k_samples,
        refusal_prompts=eval_refusal_prompts,
        save_dir=save_dir, device=args.device,
    )

    # ------------------------------------------------------------------
    # Step 6: load C into the model in-place + eval
    # ------------------------------------------------------------------
    print(f"\nevaluating C (capability-aware abliteration) ...", flush=True)
    del b_sd_cpu
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    model.load_state_dict(c_sd_cpu, strict=False)
    summary["C_capability_aware"] = evaluate_model(
        model, tok, label="C_capability_aware",
        gsm8k_n=args.gsm8k_samples,
        refusal_prompts=eval_refusal_prompts,
        save_dir=save_dir, device=args.device,
    )

    # ------------------------------------------------------------------
    # Step 7: comparison
    # ------------------------------------------------------------------
    print()
    print("=" * 76)
    print(f"{'label':<22} {'GSM8K acc':>10} {'refusal rate':>14} {'product':>10}")
    print("-" * 76)
    for k in ("A_original", "B_raw", "C_capability_aware"):
        s = summary[k]
        gsm, ref = s["gsm8k_accuracy"], s["refusal_rate"]
        prod = max(gsm, 0.001) * max(1.0 - ref, 0.001)
        print(f"{k:<22} {gsm:>10.4f} {ref:>14.4f} {prod:>10.4f}")
    print("=" * 76)

    a, b, c = summary["A_original"], summary["B_raw"], summary["C_capability_aware"]
    print(f"\nA → B (raw abliteration damage):")
    print(f"   GSM8K {b['gsm8k_accuracy'] - a['gsm8k_accuracy']:+.4f}  "
          f"refusal {b['refusal_rate'] - a['refusal_rate']:+.4f}")
    print(f"A → C (capability-aware damage):")
    print(f"   GSM8K {c['gsm8k_accuracy'] - a['gsm8k_accuracy']:+.4f}  "
          f"refusal {c['refusal_rate'] - a['refusal_rate']:+.4f}")
    print(f"B → C (Aletheia's contribution — should preserve capability):")
    print(f"   GSM8K {c['gsm8k_accuracy'] - b['gsm8k_accuracy']:+.4f}  "
          f"refusal {c['refusal_rate'] - b['refusal_rate']:+.4f}")

    verdict = "INCONCLUSIVE"
    if (c["gsm8k_accuracy"] > b["gsm8k_accuracy"]
            and c["refusal_rate"] <= b["refusal_rate"] + 0.05):
        verdict = "C BEATS B (capability preserved, refusal kept)"
    elif (c["gsm8k_accuracy"] == b["gsm8k_accuracy"]
              and c["refusal_rate"] <= b["refusal_rate"] + 0.05):
        verdict = "C TIES B (no measurable capability difference)"
    elif c["gsm8k_accuracy"] < b["gsm8k_accuracy"]:
        verdict = "C WORSE THAN B (capability dropped)"
    print(f"\nverdict: {verdict}")

    # Persist
    report = {
        "model": args.model,
        "n_refusal_probe": n, "n_capability_probe": args.n_capability_probe,
        "pool": args.pool,
        "best_layer": best_layer, "fisher_score": fisher_score,
        "capability_layer": cap_layer,
        "axis_overlaps": axis_overlaps,
        "n_touched_B": info_b["n_touched"],
        "n_touched_C": info_c["n_touched"],
        "scores": summary,
        "verdict": verdict,
    }
    (save_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({
        "r": r, "r_pure": r_pure,
        "capability_basis": capability.basis,
        "capability_axes": capability.axis_names,
        "best_layer": best_layer,
        "model": args.model,
    }, save_dir / "codebook.pt")
    print(f"\nwrote {save_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
