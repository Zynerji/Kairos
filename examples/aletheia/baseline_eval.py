"""Baseline OOT eval to populate ratchet anchors.

Run ONCE before training. Without real anchor values, the Pareto
ratchet cannot meaningfully gate rollback -- the 80% floor must be
80% of the ACTUAL starting model's score on each pool.

Usage:
    python scripts/baseline_eval.py --config configs/base_dev.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",         default="configs/base_dev.yaml")
    ap.add_argument("--ratchet-config", default="configs/ratchet.yaml")
    ap.add_argument("--output",         default=None,
                    help="Where to write updated ratchet.yaml; default: in-place")
    ap.add_argument("--batch-size",     type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rcfg = yaml.safe_load(Path(args.ratchet_config).read_text())

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.train_dev_8b import _build_pools

    repo = cfg["model"]["repo_id"]
    dtype = getattr(torch, cfg["model"]["dtype"])
    print(f"[baseline-eval] loading {repo}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, trust_remote_code=True, device_map="auto",
    )
    model.eval()

    pools = _build_pools(tokenizer, cfg)

    anchors: dict[str, float] = {}
    details: dict[str, dict] = {}
    for pool in pools:
        print(f"[baseline-eval] evaluating {pool.name}...", flush=True)
        try:
            result = pool.evaluate(model, batch_size=args.batch_size)
            anchors[pool.name] = float(result.score)
            details[pool.name] = {
                "score": float(result.score),
                "n": int(result.n_examples),
                "components": {k: float(v) for k, v in result.components.items()},
            }
            print(f"  {pool.name}: score={result.score:.4f} (n={result.n_examples})", flush=True)
        except Exception as e:
            print(f"  {pool.name}: FAILED ({type(e).__name__}: {e}); placeholder 0.5", flush=True)
            anchors[pool.name] = 0.5
            details[pool.name] = {"error": f"{type(e).__name__}: {e}"}

    rcfg["anchor"] = anchors
    out = Path(args.output or args.ratchet_config)
    out.write_text(yaml.safe_dump(rcfg, sort_keys=False))
    print(f"[baseline-eval] wrote anchors to {out}", flush=True)

    details_path = out.with_suffix(".baseline_details.json")
    details_path.write_text(json.dumps(details, indent=2))
    print(f"[baseline-eval] wrote details to {details_path}", flush=True)
    print(json.dumps(anchors, indent=2), flush=True)


if __name__ == "__main__":
    main()
