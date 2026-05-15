"""Distillation pool -- teacher imitation.

Train corpus: OpenThoughts-114k (primary), Bespoke-Stratos, s1K.
Eval: held-out slice of the same distribution -- score = normalized
sigmoid of per-token NLL improvement vs. anchor.

All MIT/Apache. No frontier API distillation. Refusal-filtered via
aletheia.distill.teacher_filter.RefusalFilter before ingestion.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator

from kairos.aletheia.pools.base import PoolBatch, PoolEvalResult
from kairos.aletheia.pools.hf_base import HFCausalLMPool
from kairos.aletheia.distill.teacher_filter import RefusalFilter


@dataclass
class DistillationPool(HFCausalLMPool):
    train_dataset_id: str = "open-thoughts/OpenThoughts-114k"
    train_subset: str | None = None
    eval_dataset_id: str = "simplescaling/s1K"
    eval_subset: str | None = None
    eval_split: str = "train"
    max_len: int = 4096
    max_new_tokens: int = 128
    refusal_filter_on: bool = True
    anchor_nll: float = 2.0

    @property
    def name(self) -> str:
        return "distillation"

    def __post_init__(self):
        self._refusal = RefusalFilter() if self.refusal_filter_on else None

    def _extract_turns(self, ex: dict) -> tuple[str, str]:
        convs = ex.get("conversations")
        if isinstance(convs, list) and convs:
            human = ""
            gpt = ""
            for c in convs:
                frm = c.get("from") or c.get("role", "")
                val = c.get("value") or c.get("content", "")
                if frm in ("human", "user") and not human:
                    human = val
                elif frm in ("gpt", "assistant") and not gpt:
                    gpt = val
            return human, gpt
        msgs = ex.get("messages")
        if isinstance(msgs, list) and msgs:
            h = next((m.get("content", "") for m in msgs if m.get("role") != "assistant"), "")
            g = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            return h, g
        return (ex.get("problem") or ex.get("prompt") or "",
                ex.get("solution") or ex.get("response") or ex.get("answer") or "")

    def _format_example(self, ex: dict) -> tuple[str, str]:
        human, gpt = self._extract_turns(ex)
        return f"{human}\n", f"{gpt}"

    def train_loader(self, batch_size: int) -> Iterator[PoolBatch]:
        ds = self._load_hf(self.train_dataset_id, self.train_subset, self.train_split)
        buf: list[dict] = []
        while True:
            for ex in ds:
                _, gpt = self._extract_turns(ex)
                if self._refusal and self._refusal.is_refusal(gpt):
                    continue
                buf.append(self._tokenize_pair(*self._format_example(ex)))
                if len(buf) >= batch_size:
                    b = self._collate(buf[:batch_size])
                    yield PoolBatch(
                        inputs={"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]},
                        targets=b["labels"],
                    )
                    buf = buf[batch_size:]

    def evaluate(self, model: Any, batch_size: int = 8) -> PoolEvalResult:
        """Score = sigmoid(anchor_nll - measured_nll) ∈ [0, 1]."""
        import math
        import torch
        import torch.nn.functional as F

        nll_total = 0.0
        tok_total = 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in self.oot_loader(batch_size):
                    inputs = {k: v.to(model.device) for k, v in batch.inputs.items()}
                    labels = batch.targets.to(model.device)
                    out = model(**inputs)
                    ce = F.cross_entropy(
                        out.logits.view(-1, out.logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    nll_total += float(ce.item())
                    tok_total += int((labels != -100).sum().item())
                    if tok_total > 5000:  # cap for cycle time
                        break
        finally:
            if was_training:
                model.train()

        if tok_total == 0:
            return PoolEvalResult(self.name, 0.0, {"nll": float("inf"), "tokens": 0}, 0)
        nll = nll_total / tok_total
        score = 1.0 / (1.0 + math.exp(nll - self.anchor_nll))
        return PoolEvalResult(
            self.name, float(score),
            {"nll": float(nll), "anchor_nll": self.anchor_nll, "tokens": tok_total},
            tok_total,
        )
