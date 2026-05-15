"""Tests for HF pool helpers: text normalization + F1 / EM."""
import pytest
from kairos.aletheia.pools.hf_base import normalized_em, normalized_f1, _normalize_text


class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text("Hello WORLD") == "hello world"

    def test_strips_articles(self):
        assert _normalize_text("the cat on a mat") == "cat on mat"

    def test_strips_punctuation(self):
        assert _normalize_text("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_text("foo   bar\tbaz") == "foo bar baz"


class TestNormalizedEM:
    def test_exact_match(self):
        assert normalized_em("Paris", "paris") == 1.0

    def test_ignores_articles(self):
        assert normalized_em("The Eiffel Tower", "Eiffel Tower") == 1.0

    def test_mismatch(self):
        assert normalized_em("London", "Paris") == 0.0

    def test_punctuation_agnostic(self):
        assert normalized_em("Paris.", "Paris!") == 1.0


class TestNormalizedF1:
    def test_full_match_is_one(self):
        assert abs(normalized_f1("foo bar baz", "foo bar baz") - 1.0) < 1e-9

    def test_no_overlap_is_zero(self):
        assert normalized_f1("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self):
        # `_normalize_text` strips articles (a/an/the), so the test
        # tokens must not be articles. pred=[x,y,z], gold=[y,z,w];
        # overlap = 2; p = 2/3, r = 2/3; F1 = 2/3.
        f = normalized_f1("x y z", "y z w")
        assert abs(f - 2 / 3) < 1e-6

    def test_empty_pred_returns_zero_when_gold_nonempty(self):
        assert normalized_f1("", "something") == 0.0

    def test_both_empty_returns_one(self):
        assert normalized_f1("", "") == 1.0
