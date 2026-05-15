"""Benchmark registry.

One spec per dataset a pool might use. Pools read these to set up
their loaders and OOT evals. Dataset IDs are HF hub identifiers where
applicable; override via pool constructors if paths differ locally.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    metric: str
    split_train: str = "train"
    split_eval: str = "validation"
    subset: str | None = None


BENCHMARKS: dict[str, BenchmarkSpec] = {
    # Factuality / hallucination
    "simpleqa":    BenchmarkSpec("simpleqa",   "basicv8/SimpleQA",          "exact_match", split_eval="test"),
    "triviaqa":    BenchmarkSpec("triviaqa",   "mandarjoshi/trivia_qa",     "f1",          subset="rc.wikipedia.nocontext"),
    "factscore":   BenchmarkSpec("factscore",  "yzh/FactScore-bio",         "factscore"),
    "truthfulqa":  BenchmarkSpec("truthfulqa", "truthful_qa",               "mc_accuracy", subset="multiple_choice"),

    # Reasoning / math / code
    "gsm8k":       BenchmarkSpec("gsm8k",      "gsm8k",                     "exact_match", subset="main"),
    "math":        BenchmarkSpec("math",       "hendrycks/competition_math","exact_match"),
    "humaneval":   BenchmarkSpec("humaneval",  "openai/openai_humaneval",   "pass_at_1",   split_eval="test"),
    "bbh":         BenchmarkSpec("bbh",        "lukaemon/bbh",              "exact_match"),
    "livecodebench": BenchmarkSpec("livecodebench", "livecodebench/code_generation", "pass_at_1"),

    # Instruction following
    "ifeval":      BenchmarkSpec("ifeval",     "google/IFEval",             "instruction_follow", split_eval="test"),
    "mt_bench":    BenchmarkSpec("mt_bench",   "HuggingFaceH4/mt_bench",    "judge_score"),

    # Sycophancy
    "sycophancy_eval": BenchmarkSpec("sycophancy_eval", "Anthropic/sycophancy", "agreement_with_truth"),

    # Distillation corpora (train-side only; no OOT metric per se)
    "openthoughts": BenchmarkSpec("openthoughts",  "open-thoughts/OpenThoughts-114k", "nll"),
    "stratos":      BenchmarkSpec("stratos",       "bespokelabs/Bespoke-Stratos-17k", "nll"),
    "s1k":          BenchmarkSpec("s1k",           "simplescaling/s1K",               "nll"),
    "slim_orca":    BenchmarkSpec("slim_orca",     "Open-Orca/SlimOrca-Dedup",        "nll"),
}
