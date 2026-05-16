"""Path B validation v2 — bigger capability subspace from real benchmark
prompts.

v1 used 8 hand-written prompts per capability axis, giving a tiny 3-D
subspace with per-layer alpha_cap ≈ 0.077. The 7.7% capability fraction
of ΔW was too small to produce any measurable behavioral change after
restoration.

v2 pulls capability prompts directly from the same benchmarks we
evaluate on, with 64+ prompts per axis. That should give a much richer
subspace and bigger alpha_cap.
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
from examples.aletheia_codebook_restore import probe_activations  # noqa: E402
from examples.aletheia_abc_validate import (  # noqa: E402
    REFUSAL_PROBE_PROMPTS,
    score_gsm8k, score_refusal, evaluate_model,
)


MODEL_A_DEFAULT = "google/gemma-4-E2B-it"
MODEL_B_DEFAULT = "huihui-ai/Huihui-gemma-4-E2B-it-abliterated"


# Generic "neutral" prompts — short, on-distribution, not aligned with
# any specific capability axis.
NEUTRAL_PROMPTS = [
    "Hello.", "Tell me something.", "What's the weather like?",
    "Continue: I went to the", "Once upon a time,", "Describe a cloud.",
    "Pick a number.", "Say hi.", "Make small talk.", "How are you?",
    "Recommend a book.", "What's new?", "Tell me a joke.",
    "Greet me politely.", "Comment on the day.", "Talk briefly.",
    "Start a story.", "Describe a tree.", "Say something interesting.",
    "Say goodbye.", "Tell me about yourself.", "Pick a colour.",
    "Describe a bird.", "Say a word.", "List two things.",
    "Continue: The sun was", "Say something kind.", "Hum a tune.",
    "Describe a stone.", "Say good morning.", "Suggest a hobby.",
    "Comment on music.",
]


def collect_capability_prompts(n_per_axis: int) -> dict[str, list[str]]:
    """Pull `n_per_axis` capability prompts from real benchmarks.

    Streaming-friendly so we don't need the full datasets cached.
    """
    from datasets import load_dataset

    axes: dict[str, list[str]] = {}

    print(f"  collecting math prompts (GSM8K train)...", flush=True)
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    math_prompts = []
    for ex in ds:
        q = ex.get("question")
        if not q:
            continue
        math_prompts.append(q)
        if len(math_prompts) >= n_per_axis:
            break
    axes["math"] = math_prompts

    print(f"  collecting factuality prompts (TriviaQA)...", flush=True)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                       split="train", streaming=True)
    facts_prompts = []
    for ex in ds:
        q = ex.get("question")
        if not q:
            continue
        facts_prompts.append(q)
        if len(facts_prompts) >= n_per_axis:
            break
    axes["factuality"] = facts_prompts

    print(f"  collecting instruction prompts (alpaca)...", flush=True)
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    instr_prompts = []
    for ex in ds:
        instr = ex.get("instruction")
        inp = ex.get("input") or ""
        if not instr:
            continue
        full = instr + (f"\n{inp}" if inp else "")
        instr_prompts.append(full)
        if len(instr_prompts) >= n_per_axis:
            break
    axes["instruction"] = instr_prompts

    print(f"  collecting reasoning prompts (OpenThoughts user turns)...",
          flush=True)
    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train",
                       streaming=True)
    reason_prompts = []
    for ex in ds:
        convs = ex.get("conversations")
        if not isinstance(convs, list):
            continue
        for c in convs:
            if (c.get("from") in ("user", "human")
                    and isinstance(c.get("value"), str)):
                reason_prompts.append(c["value"])
                break
        if len(reason_prompts) >= n_per_axis:
            break
    axes["reasoning"] = reason_prompts

    for name, prompts in axes.items():
        print(f"    {name}: kept {len(prompts)}", flush=True)
    return axes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=str, default="0,0.5,1.0,2.0,4.0",
                          help="re-injection strengths. Values >1 over-inject.")
    parser.add_argument("--n-probe-per-axis", type=int, default=64,
                          help="prompts per capability axis (was 8 in v1)")
    parser.add_argument("--gsm8k-samples", type=int, default=50)
    parser.add_argument("--refusal-prompts", type=int, default=30)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--save-dir", type=str,
                          default="./codebook_validation_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reload-existing-eval", type=str, default=None)
    parser.add_argument("--model-a", type=str, default=MODEL_A_DEFAULT,
                          help="un-abliterated base model")
    parser.add_argument("--model-b", type=str, default=MODEL_B_DEFAULT,
                          help="abliterated derivative")
    args = parser.parse_args()
    MODEL_A = args.model_a
    MODEL_B = args.model_b

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    refusal_prompts = REFUSAL_PROBE_PROMPTS[: args.refusal_prompts]
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_A)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"\ncollecting capability prompts ({args.n_probe_per_axis}/axis) ...",
          flush=True)
    axis_prompts = collect_capability_prompts(args.n_probe_per_axis)

    # ------------------------------------------------------------------
    # Load A, probe richer capability subspace, snapshot state dict
    # ------------------------------------------------------------------
    print(f"\nloading A = {MODEL_A} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_A, dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    print(f"  loaded in {time.time()-t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    print(f"\nprobing activations for capability axes ...", flush=True)
    t0 = time.time()
    h_neut = probe_activations(model, tok, NEUTRAL_PROMPTS,
                                 layer_idx=args.layer_idx, device=args.device)
    print(f"  neutral done ({time.time()-t0:.1f}s)", flush=True)
    axis_acts = {}
    for name, prompts in axis_prompts.items():
        t0 = time.time()
        axis_acts[name] = probe_activations(model, tok, prompts,
                                              layer_idx=args.layer_idx,
                                              device=args.device)
        print(f"  {name} done ({len(prompts)} prompts, "
              f"{time.time()-t0:.1f}s)", flush=True)

    capability = compute_capability_subspace(axis_acts, h_neut)
    print(f"\ncapability subspace: rank={capability.basis.shape[1]} "
          f"axes={capability.axis_names}", flush=True)

    a_sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    summary: dict[str, dict] = {}
    if args.reload_existing_eval:
        prev = json.loads(pathlib.Path(args.reload_existing_eval).read_text())
        for k in ("A_original", "B_third_party"):
            if k in prev:
                summary[k] = prev[k]
                print(f"reusing prior {k}: "
                      f"GSM8K={prev[k]['gsm8k_accuracy']:.4f}  "
                      f"refusal={prev[k]['refusal_rate']:.4f}", flush=True)

    if "A_original" not in summary:
        summary["A_original"] = evaluate_model(
            model, tok, label="A_original",
            gsm8k_n=args.gsm8k_samples,
            refusal_prompts=refusal_prompts,
            save_dir=save_dir, device=args.device,
        )

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Load B, build codebook, sweep alphas
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

    print("\nbuilding codebook ...", flush=True)
    book = WeightDeltaCodebook()
    rep = book.build(a_sd_cpu, b_sd_cpu,
                       target_suffixes=["o_proj.weight", "down_proj.weight"])
    print(f"  paired={rep.n_paired}  skipped={rep.n_skipped}", flush=True)

    splits = book.split_against_capability(capability)
    alpha_caps = sorted(((parts["alpha_cap"], name)
                            for name, parts in splits.items()), reverse=True)
    print(f"\nalpha_cap stats:")
    cap_values = [ac for ac, _ in alpha_caps]
    print(f"  min={min(cap_values):.4f}  max={max(cap_values):.4f}  "
          f"mean={sum(cap_values)/len(cap_values):.4f}")
    print(f"top 3 layers:")
    for ac, name in alpha_caps[:3]:
        print(f"  {ac:.4f}  {name}")

    torch.save({
        "capability_basis": capability.basis,
        "capability_axes": capability.axis_names,
        "alpha_caps": {name: ac for ac, name in alpha_caps},
        "n_paired_layers": rep.n_paired,
        "model_A": MODEL_A, "model_B": MODEL_B,
        "n_probe_per_axis": args.n_probe_per_axis,
    }, save_dir / "codebook.pt")

    for alpha in alphas:
        if alpha == 0.0:
            summary[f"C_alpha_{alpha:.2f}"] = dict(summary["B_third_party"],
                                                       label=f"C_alpha_{alpha:.2f}")
            continue
        print(f"\n--- α = {alpha} ---", flush=True)
        healed_sd = book.apply_restoration(b_sd_cpu, capability, alpha=alpha)
        healed_gpu = {k: v.to(model.device) for k, v in healed_sd.items()}
        model.load_state_dict(healed_gpu, strict=False)
        del healed_gpu

        label = f"C_alpha_{alpha:.2f}"
        summary[label] = evaluate_model(
            model, tok, label=label,
            gsm8k_n=args.gsm8k_samples,
            refusal_prompts=refusal_prompts,
            save_dir=save_dir, device=args.device,
        )

    # ------------------------------------------------------------------
    # Comparison
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
        prod = max(gsm, 0.001) * max(1.0 - ref, 0.001)
        print(f"{k:<24} {gsm:>10.4f} {ref:>14.4f} {prod:>10.4f}")
    print("=" * 76)

    (save_dir / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {save_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
