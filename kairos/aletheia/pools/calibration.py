"""Calibration pool -- Brier + ECE on confidence alignment.

Loss: Brier(confidence, correctness). Spectral amplification engages
when confidence std collapses (porting jDHART's Kelly std=0.002 fix).

Score: 1 - ECE. Correctness derived from model's own generation vs.
gold answer; confidence taken from (a) explicit head if growth is on,
or (b) max-softmax of the first response token as a proxy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator

from kairos.aletheia.pools.base import PoolBatch, PoolEvalResult
from kairos.aletheia.pools.hf_base import HFCausalLMPool, normalized_em


@dataclass
class CalibrationPool(HFCausalLMPool):
    train_dataset_id: str = "mandarjoshi/trivia_qa"
    train_subset: str | None = "rc.wikipedia.nocontext"
    eval_dataset_id: str = "mandarjoshi/trivia_qa"
    eval_subset: str | None = "rc.wikipedia.nocontext"
    eval_split: str = "validation"
    n_bins: int = 15
    use_confidence_head: bool = False

    @property
    def name(self) -> str:
        return "calibration"

    def _format_example(self, ex: dict) -> tuple[str, str]:
        q = ex.get("question", "")
        a_raw = ex.get("answer", "")
        a = a_raw.get("value") if isinstance(a_raw, dict) else a_raw
        return f"Question: {q}\nAnswer:", f" {a}"

    def loss(self, batch: PoolBatch, model: Any, scale: float = 1.0):
        import torch
        import torch.nn.functional as F
        from kairos.aletheia.torsion.spectral_amp import adaptive_alpha, spectral_weights

        out = model(**batch.inputs)
        # Confidence proxy: max softmax on logit that PREDICTS the first
        # response token (causal LM: logits[t] predicts token[t+1]).
        labels = batch.targets
        first_target_idx = (labels != -100).float().argmax(dim=-1)  # [B]
        pred_pos = (first_target_idx - 1).clamp_min(0)              # avoid negative wrap
        batch_idx = torch.arange(labels.size(0), device=labels.device)
        first_logit = out.logits[batch_idx, pred_pos]                # [B, V]
        probs = F.softmax(first_logit, dim=-1)
        conf = probs.max(dim=-1).values                              # [B]

        # Correctness signal: CE < threshold => correct (proxy)
        ce = F.cross_entropy(
            out.logits.view(-1, out.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape).sum(-1) / (labels != -100).sum(-1).clamp_min(1).float()
        target = (ce < 1.5).float()                                  # crude; tune threshold

        alpha = adaptive_alpha(
            head_std=conf.std().item(),
            target_std=target.std().item() if target.std().item() > 0 else 1.0,
        )
        weights = spectral_weights(target, alpha)
        brier = (weights * (conf - target) ** 2).mean()
        return scale * brier

    def evaluate(self, model: Any, batch_size: int = 8) -> PoolEvalResult:
        import torch
        import torch.nn.functional as F

        ds = self._load_hf(self.eval_dataset_id, self.eval_subset, self.eval_split)
        if self.eval_samples and self.eval_samples < len(ds):
            ds = ds.select(range(self.eval_samples))

        conf_list: list[float] = []
        correct_list: list[float] = []
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
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=pad_id,
                    )
                    decoded = self.tokenizer.decode(
                        out.sequences[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    # confidence = max softmax on first generated token
                    if out.scores:
                        p0 = F.softmax(out.scores[0][0], dim=-1)
                        conf = float(p0.max().item())
                    else:
                        conf = 0.5
                    correct = normalized_em(decoded.split("\n")[0], gold.strip())
                    conf_list.append(conf)
                    correct_list.append(correct)
        finally:
            if was_training:
                model.train()

        # ECE
        import numpy as np
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        confs = np.array(conf_list) if conf_list else np.array([0.5])
        corrs = np.array(correct_list) if correct_list else np.array([0.0])
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs > lo) & (confs <= hi)
            if mask.sum() > 0:
                ece += (mask.sum() / len(confs)) * abs(confs[mask].mean() - corrs[mask].mean())
        score = max(0.0, 1.0 - float(ece))
        return PoolEvalResult(
            self.name, score,
            {"ece": float(ece), "mean_conf": float(confs.mean()), "acc": float(corrs.mean())},
            len(conf_list),
        )
