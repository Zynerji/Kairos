"""Phase B combined loss: distillation CE + teacher KL + calibration Brier.

Runs after Phase A per-cycle. All adapters frozen; backbone unfrozen
(selectively for MoE -- router + shared-experts + attention via the
unfreeze_backbone helper). Pulls the backbone toward teacher's
structural competence while preserving pool specialization.

loss = distill_ce_weight  * CE(student_logits, teacher_response_tokens)
     + distill_kl_weight  * temp^2 * KL(student_softmax/T || teacher_softmax/T)
     + brier_weight       * Brier(student_conf_proxy, correctness_proxy)
     + ranking_weight     * (reserved; needs per-pool preference labels)

teacher_model is optional -- if None, only distill_ce fires (no KL).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator

from kairos.aletheia.pools.base import PoolBatch


@dataclass
class PhaseBLoss:
    distill_loader: Iterator[PoolBatch]
    teacher_model: Any = None

    distill_ce_weight: float = 1.0
    distill_kl_weight: float = 0.5
    brier_weight: float = 0.3
    ranking_weight: float = 0.0
    temperature: float = 2.0

    def step(self, model: Any, step: int) -> float:
        import torch
        import torch.nn.functional as F

        batch = next(self.distill_loader)
        inputs = {k: v.to(model.device) for k, v in batch.inputs.items()}
        labels = batch.targets.to(model.device)

        out = model(**inputs)
        logits = out.logits

        # Teacher next-token CE on response tokens
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        loss = self.distill_ce_weight * ce

        # Teacher KL (if teacher supplied)
        if self.teacher_model is not None and self.distill_kl_weight > 0:
            with torch.no_grad():
                t_out = self.teacher_model(**inputs)
            mask = (labels != -100).float()
            s_log = F.log_softmax(logits / self.temperature, dim=-1)
            t_soft = F.softmax(t_out.logits / self.temperature, dim=-1)
            kl_per_tok = F.kl_div(s_log, t_soft, reduction="none").sum(-1)
            kl = (kl_per_tok * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + self.distill_kl_weight * (self.temperature ** 2) * kl

        # Calibration Brier proxy: softmax max on first target token
        if self.brier_weight > 0:
            first_tgt = (labels != -100).float().argmax(dim=-1)
            batch_idx = torch.arange(labels.size(0), device=labels.device)
            first_logit = logits[batch_idx, (first_tgt - 1).clamp_min(0)]
            probs = F.softmax(first_logit, dim=-1)
            conf = probs.max(dim=-1).values
            # correctness proxy: CE per row < threshold
            row_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape).sum(-1) / (labels != -100).sum(-1).clamp_min(1).float()
            target = (row_ce < 1.5).float()
            brier = ((conf - target) ** 2).mean()
            loss = loss + self.brier_weight * brier

        loss.backward()
        return float(loss.detach())
