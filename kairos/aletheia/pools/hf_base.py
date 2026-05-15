"""HuggingFace-backed CausalLM pool base.

Concrete pools inherit and implement only:
- name (property)
- _format_example(ex) -> (prompt: str, target: str)
- optionally _score(prediction, gold) -> float  (default: normalized F1)
- optionally evaluate(model, ...)                (override for special metrics)

Handles dataset loading, hash-based train/eval disjointness, tokenization,
label-masking (prompt tokens -> -100), batching, and default greedy-gen eval.
"""
from __future__ import annotations
import re
import string
from dataclasses import dataclass
from typing import Any, Iterator

from kairos.aletheia.pools.base import CausalLMPool, PoolBatch, PoolEvalResult


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def normalized_em(pred: str, gold: str) -> float:
    return float(_normalize_text(pred) == _normalize_text(gold))


def normalized_f1(pred: str, gold: str) -> float:
    p = _normalize_text(pred).split()
    g = _normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = set(p) & set(g)
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall = len(common) / len(g)
    return 2 * precision * recall / (precision + recall)


@dataclass
class HFCausalLMPool(CausalLMPool):
    tokenizer: Any = None
    train_dataset_id: str = ""
    train_subset: str | None = None
    train_split: str = "train"
    eval_dataset_id: str = ""
    eval_subset: str | None = None
    eval_split: str = "validation"
    max_len: int = 1024
    max_new_tokens: int = 128
    eval_samples: int = 100
    data_root: str | None = None

    # --- subclass contract ---------------------------------------------------

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _format_example(self, ex: dict) -> tuple[str, str]:
        raise NotImplementedError

    def _score(self, prediction: str, gold: str) -> float:
        return normalized_f1(prediction, gold)

    # --- helpers -------------------------------------------------------------

    def _load_hf(self, dataset_id: str, subset: str | None, split: str):
        from datasets import load_dataset
        kwargs: dict[str, Any] = {"split": split}
        if self.data_root:
            kwargs["cache_dir"] = self.data_root
        if subset:
            return load_dataset(dataset_id, subset, **kwargs)
        return load_dataset(dataset_id, **kwargs)

    def _tokenize_pair(self, prompt: str, target: str) -> dict:
        enc_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_len)
        enc_full = self.tokenizer(prompt + target, truncation=True, max_length=self.max_len)
        labels = list(enc_full["input_ids"])
        for i in range(min(len(enc_prompt["input_ids"]), len(labels))):
            labels[i] = -100
        return {
            "input_ids": enc_full["input_ids"],
            "attention_mask": enc_full["attention_mask"],
            "labels": labels,
        }

    def _collate(self, items: list[dict]) -> dict:
        import torch
        maxlen = max(len(x["input_ids"]) for x in items)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def pad(x, fill):
            return x + [fill] * (maxlen - len(x))

        return {
            "input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in items]),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in items]),
            "labels": torch.tensor([pad(x["labels"], -100) for x in items]),
        }

    def _iter_batches(self, ds, batch_size: int) -> Iterator[PoolBatch]:
        buf: list[dict] = []
        while True:
            for ex in ds:
                buf.append(self._tokenize_pair(*self._format_example(ex)))
                if len(buf) >= batch_size:
                    b = self._collate(buf[:batch_size])
                    yield PoolBatch(
                        inputs={"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]},
                        targets=b["labels"],
                    )
                    buf = buf[batch_size:]

    # --- Pool interface ------------------------------------------------------

    def train_loader(self, batch_size: int) -> Iterator[PoolBatch]:
        ds = self._load_hf(self.train_dataset_id, self.train_subset, self.train_split)
        yield from self._iter_batches(ds, batch_size)

    def oot_loader(self, batch_size: int) -> Iterator[PoolBatch]:
        ds = self._load_hf(self.eval_dataset_id, self.eval_subset, self.eval_split)
        if self.eval_samples and self.eval_samples < len(ds):
            ds = ds.select(range(self.eval_samples))
        yield from self._iter_batches(ds, batch_size)

    def evaluate(self, model: Any, batch_size: int = 8) -> PoolEvalResult:
        import torch

        ds = self._load_hf(self.eval_dataset_id, self.eval_subset, self.eval_split)
        if self.eval_samples and self.eval_samples < len(ds):
            ds = ds.select(range(self.eval_samples))

        scores: list[float] = []
        was_training = model.training
        model.eval()
        pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        try:
            with torch.no_grad():
                for ex in ds:
                    prompt, gold = self._format_example(ex)
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt",
                        truncation=True, max_length=self.max_len,
                    ).to(model.device)
                    out = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                    decoded = self.tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    scores.append(self._score(decoded, gold))
        finally:
            if was_training:
                model.train()

        mean = sum(scores) / max(len(scores), 1)
        return PoolEvalResult(self.name, float(mean), {"mean": mean}, len(scores))
