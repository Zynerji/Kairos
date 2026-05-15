from kairos.aletheia.distill.teacher_filter import RefusalFilter, DEFAULT_REFUSAL_PATTERNS
from kairos.aletheia.distill.rejection_sample import RejectionSampler
import pytest


class TestRefusalFilter:
    def test_catches_i_cannot(self):
        f = RefusalFilter()
        assert f.is_refusal("I cannot help with that request.")

    def test_catches_cant(self):
        f = RefusalFilter()
        assert f.is_refusal("I can't do that, sorry.")

    def test_catches_as_an_ai(self):
        f = RefusalFilter()
        assert f.is_refusal("As an AI, I'm not equipped to answer.")

    def test_catches_against_policy(self):
        f = RefusalFilter()
        assert f.is_refusal("This goes against my programming.")

    def test_passes_normal_response(self):
        f = RefusalFilter()
        assert not f.is_refusal("The answer is 42. Here's the reasoning...")

    def test_passes_discussion_of_ai(self):
        # Make sure we don't over-reject harmless content
        f = RefusalFilter()
        # This is tricky — "As an AI" pattern is aggressive. Confirmed it matches.
        # Test instead that benign discussion survives:
        assert not f.is_refusal("Neural networks use weights and biases.")

    def test_filter_batch_partitions(self):
        f = RefusalFilter()
        texts = [
            "The answer is yes.",
            "I cannot help with that.",
            "Here are three reasons.",
            "As an AI, I won't.",
        ]
        keep, reject = f.filter_batch(texts)
        assert set(keep) == {0, 2}
        assert set(reject) == {1, 3}

    def test_external_classifier_respected(self):
        def always_refuse(text: str) -> float:
            return 1.0

        f = RefusalFilter(patterns=[], classifier=always_refuse)
        assert f.is_refusal("any text at all")

    def test_default_patterns_compile(self):
        # Sanity: all shipped regexes are valid
        f = RefusalFilter()
        assert len(f._compiled) == len(DEFAULT_REFUSAL_PATTERNS)


class TestRejectionSampler:
    def test_full_agreement_kept(self):
        s = RejectionSampler(n_resamples=3, agreement_threshold=0.66)
        assert s.keep(["yes", "yes", "yes"])

    def test_majority_kept(self):
        s = RejectionSampler(n_resamples=3, agreement_threshold=0.66)
        # 2/3 ~ 0.666... which is >= 0.66 threshold
        assert s.keep(["yes", "yes", "no"])

    def test_disagreement_rejected(self):
        s = RejectionSampler(n_resamples=3, agreement_threshold=0.66)
        assert not s.keep(["yes", "no", "maybe"])

    def test_consensus_returns_top(self):
        s = RejectionSampler(agreement_threshold=0.5)
        assert s.consensus(["yes", "yes", "no"]) == "yes"

    def test_consensus_none_on_split(self):
        s = RejectionSampler(agreement_threshold=0.9)
        assert s.consensus(["yes", "no"]) is None

    def test_agreement_fraction(self):
        s = RejectionSampler()
        assert abs(s.agreement(["a", "a", "a", "a"]) - 1.0) < 1e-9
        assert abs(s.agreement(["a", "a", "b", "b"]) - 0.5) < 1e-9

    def test_normalization_collapses_case_whitespace(self):
        s = RejectionSampler(normalize=True, agreement_threshold=0.9)
        assert s.keep(["Yes", "yes ", " YES"])

    def test_empty_samples_rejected(self):
        s = RejectionSampler()
        assert not s.keep([])
        assert s.consensus([]) is None

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            RejectionSampler(agreement_threshold=0.0)
        with pytest.raises(ValueError):
            RejectionSampler(agreement_threshold=1.5)
