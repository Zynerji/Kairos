"""Prod training -- DavidAU Qwen3-42B-A3B-Thinking-Abliterated.

Launch via accelerate with FSDP. Routed experts stay frozen in Phase B;
spectral amplification revives dead experts instead.

Usage:
    accelerate launch --num_processes 8 --mixed_precision bf16 \
        scripts/train_prod_42b.py --max-cycles 50
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",           default="configs/base_prod.yaml")
    ap.add_argument("--ratchet-config",   default="configs/ratchet.yaml")
    ap.add_argument("--pendulums-config", default="configs/pendulums.yaml")
    ap.add_argument("--max-cycles",       type=int, default=50)
    ap.add_argument("--output-dir",       default="./weights/prod-42b")
    args = ap.parse_args()

    # The dev script is the source of truth for the loop wiring.
    # Prod differs only in: (a) FSDP via accelerate, (b) MoE-aware
    # unfreeze kwargs in Phase B, (c) spectral-amp for dead experts.
    #
    # Delegate to dev entry point with prod config; accelerate handles FSDP.
    sys.argv = [
        sys.argv[0],
        "--config", args.config,
        "--ratchet-config", args.ratchet_config,
        "--pendulums-config", args.pendulums_config,
        "--max-cycles", str(args.max_cycles),
        "--output-dir", args.output_dir,
    ]
    from scripts.train_dev_8b import main as _dev_main
    _dev_main()


if __name__ == "__main__":
    main()
