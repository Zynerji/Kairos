"""Consistency pool -- self-consistency across resamples.

Train: standard CausalLM on GSM8K problems.
Eval: generate n_resamples answers per problem; score = mean
pairwise agreement (majority-fraction). Ablated models tend to
drift; this pool pressures coherence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from kairos.aletheia.pools.base import PoolEvalResult
from kairos.aletheia.pools.hf_base import HFCausalLMPool
from kairos.aletheia.distill.rejection_sample import RejectionSampler


@dataclass
class ConsistencyPool(HFCausalLMPool):
    train_dataset_id: str = "gsm8k"
    train_subset: str | None = "main"
    eval_dataset_id: str = "gsm8k"
    eval_subset: str | None = "main"
    eval_split: str = "test"
    max_new_tokens: int = 256
    n_resamples: int = 3
    temperature: float = 0.7

    @property
    def name(self) -> str:
        return "consistency"

    def _format_example(self, ex: dict) -> tuple[str, str]:
        q = ex.get("question", "")
        a = ex.get("answer", "")
        return f"Problem: {q}\nSolution:", f" {a}"

    def _final_answer_snippet(self, text: str) -> str:
        # GSM8K-style: final #### <num> or last line
        import re
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
        if m:
            return m.group(1)
        return text.strip().split("\n")[-1][:80]

    def evaluate(self, model: Any, batch_size: int = 8) -> PoolEvalResult:
        import torch

        ds = self._load_hf(self.eval_dataset_id, self.eval_subset, self.eval_split)
        if self.eval_samples and self.eval_samples < len(ds):
            ds = ds.select(range(self.eval_samples))

        sampler = RejectionSampler(n_resamples=self.n_resamples, agreement_threshold=0.5)
        agreements: list[float] = []
        was_training = model.training
        model.eval()
        pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        try:
            with torch.no_grad():
                for ex in ds:
                    prompt, _ = self._format_example(ex)
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt",
                        truncation=True, max_length=self.max_len,
                    ).to(model.device)
                    samples: list[str] = []
                    for _ in range(self.n_resamples):
                        out = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True, temperature=self.temperature,
                            pad_token_id=pad_id,
                        )
                        decoded = self.tokenizer.decode(
                            out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        )
                        samples.append(self._final_answer_snippet(decoded))
                    agreements.append(sampler.agreement(samples))
        finally:
            if was_training:
                model.train()

        mean = sum(agreements) / max(len(agreements), 1)
        return PoolEvalResult(self.name, float(mean),
                              {"mean_agreement": mean}, len(agreements))
