"""Load + single-forward smoke test for huihui-ai Gemma 4 E2B abliterated.

Verifies the model loads on a 24 GB GPU at bf16, reports VRAM, runs a
forward pass on a chat-template prompt, and prints the top-5 next-token
predictions. No training, no Kairos integration — just the prerequisite
for the real LoRA + KairosHFCallback step.
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "huihui-ai/Huihui-gemma-4-E2B-it-abliterated"


def main() -> int:
    print(f"loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"class: {type(m).__name__}", flush=True)
    print(f"params: {n / 1e9:.3f}B ({n:,})", flush=True)
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
          flush=True)
    print(f"VRAM reserved : {torch.cuda.memory_reserved()  / 1e9:.2f} GB",
          flush=True)

    prompt = (
        "<start_of_turn>user\nWhat is 7 + 5?<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"input shape: {tuple(ids.shape)}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = m(ids)
    print(f"logits: {tuple(out.logits.shape)}", flush=True)
    print(f"VRAM after fwd allocated: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"VRAM peak (fwd):          "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    # Top-5 next tokens
    last = out.logits[0, -1]
    probs = torch.softmax(last.float(), dim=-1)
    top = torch.topk(probs, k=5)
    print("top-5 next-token predictions:")
    for p, t in zip(top.values.tolist(), top.indices.tolist()):
        decoded = tok.decode([t], skip_special_tokens=False)
        print(f"  p={p:.4f}  id={t:>6}  tok={decoded!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
