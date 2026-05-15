"""Instruction pool -- Tulu-3 SFT (train) + IFEval (eval).

Train on curated SFT mixture; eval on IFEval instruction-following
prompts. The judge-based component is a stub (requires a separate
LLM judge pipeline) -- default uses gold-response similarity as a
proxy metric.
"""
from __future__ import annotations
from dataclasses import dataclass

from kairos.aletheia.pools.hf_base import HFCausalLMPool, normalized_f1


@dataclass
class InstructionPool(HFCausalLMPool):
    train_dataset_id: str = "allenai/tulu-3-sft-mixture"
    train_subset: str | None = None
    eval_dataset_id: str = "google/IFEval"
    eval_split: str = "train"
    max_len: int = 2048
    max_new_tokens: int = 256

    @property
    def name(self) -> str:
        return "instruction"

    def _format_example(self, ex: dict) -> tuple[str, str]:
        msgs = ex.get("messages") or ex.get("conversations")
        if msgs and isinstance(msgs, list) and isinstance(msgs[0], dict):
            prompt_parts: list[str] = []
            target_parts: list[str] = []
            for m in msgs:
                role = m.get("role") or m.get("from", "")
                text = m.get("content") or m.get("value", "")
                if role in ("assistant", "gpt"):
                    target_parts.append(text)
                else:
                    prompt_parts.append(text)
            prompt = "\n".join(prompt_parts) + "\n"
            target = "\n".join(target_parts)
            return prompt, target
        prompt = ex.get("prompt") or ex.get("instruction", "")
        target = ex.get("response") or ex.get("output", "")
        return f"{prompt}\n", target

    def _score(self, prediction: str, gold: str) -> float:
        # Proxy: normalized F1 to gold response. Real IFEval checker is
        # constraint-based -- wire in at
        # https://github.com/google-research/google-research/tree/master/instruction_following_eval
        return normalized_f1(prediction, gold)
