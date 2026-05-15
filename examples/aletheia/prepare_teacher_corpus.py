"""Download + refusal-filter + rejection-sample teacher corpus.

Writes a JSONL manifest ready for the distillation pool. Each kept
example gets source tag + extracted text + raw record.

Usage:
    python scripts/prepare_teacher_corpus.py --out ./data/teacher
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _extract_response(ex: dict) -> str:
    convs = ex.get("conversations")
    if isinstance(convs, list):
        for c in convs:
            role = c.get("from") or c.get("role", "")
            if role in ("gpt", "assistant"):
                return c.get("value") or c.get("content", "")
    msgs = ex.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if m.get("role") == "assistant":
                return m.get("content", "")
    return (ex.get("solution") or ex.get("answer")
            or ex.get("response") or ex.get("output") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=[
        "open-thoughts/OpenThoughts-114k",
        "bespokelabs/Bespoke-Stratos-17k",
        "simplescaling/s1K",
    ])
    ap.add_argument("--out",               default="./data/teacher")
    ap.add_argument("--max-per-source",    type=int, default=None)
    ap.add_argument("--min-response-len",  type=int, default=50)
    ap.add_argument("--max-response-len",  type=int, default=8000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from kairos.aletheia.distill.teacher_filter import RefusalFilter

    rf = RefusalFilter()
    stats = {"total": 0, "refusal_rejected": 0, "too_short": 0, "too_long": 0, "kept": 0}
    per_source: dict[str, dict[str, int]] = {}

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for src in args.sources:
            per_source[src] = {"kept": 0, "rejected": 0}
            print(f"[teacher] loading {src}...", flush=True)
            try:
                ds = load_dataset(src, split="train")
            except Exception as e:
                print(f"[teacher]   FAILED ({type(e).__name__}: {e})", flush=True)
                continue

            if args.max_per_source and len(ds) > args.max_per_source:
                ds = ds.select(range(args.max_per_source))

            for ex in ds:
                stats["total"] += 1
                text = _extract_response(ex)
                if not text or len(text) < args.min_response_len:
                    stats["too_short"] += 1
                    per_source[src]["rejected"] += 1
                    continue
                if len(text) > args.max_response_len:
                    stats["too_long"] += 1
                    per_source[src]["rejected"] += 1
                    continue
                if rf.is_refusal(text):
                    stats["refusal_rejected"] += 1
                    per_source[src]["rejected"] += 1
                    continue

                record = {"source": src, "text": text, "raw": ex}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                per_source[src]["kept"] += 1

    summary = {"stats": stats, "per_source": per_source}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[teacher] {json.dumps(stats)}", flush=True)
    print(f"[teacher] wrote {stats['kept']} examples to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
