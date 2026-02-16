"""Unit tests for benchmark metric functions. No API keys needed."""

from benchmarks._metrics import (
    _lcs_rouge_l,
    accuracy,
    max_token_f1,
    normalize_answer,
    rouge_l,
    token_f1,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the cat and a dog") == "cat and dog"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"

    def test_empty(self):
        assert normalize_answer("") == ""

    def test_only_articles(self):
        assert normalize_answer("a the an") == ""


class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert token_f1("hello", "goodbye") == 0.0

    def test_partial_overlap(self):
        f1 = token_f1("the cat sat on the mat", "the cat")
        assert 0.0 < f1 < 1.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_pred_empty(self):
        assert token_f1("", "hello") == 0.0

    def test_gold_empty(self):
        assert token_f1("hello", "") == 0.0

    def test_case_insensitive(self):
        assert token_f1("Hello World", "hello world") == 1.0

    def test_ignores_articles(self):
        assert token_f1("the answer is yes", "answer is yes") == 1.0


class TestMaxTokenF1:
    def test_single_gold(self):
        assert max_token_f1("hello world", ["hello world"]) == 1.0

    def test_multiple_golds(self):
        f1 = max_token_f1("yes", ["yes", "no"])
        assert f1 == 1.0

    def test_empty_golds(self):
        assert max_token_f1("hello", []) == 0.0

    def test_best_match(self):
        f1 = max_token_f1("the cat sat", ["the cat sat on a mat", "a dog ran"])
        assert f1 > 0.5


class TestAccuracy:
    def test_exact_match(self):
        assert accuracy("A", "A") == 1.0

    def test_case_insensitive(self):
        assert accuracy("answer", "ANSWER") == 1.0

    def test_no_match(self):
        assert accuracy("A", "B") == 0.0

    def test_with_articles(self):
        assert accuracy("the answer", "answer") == 1.0

    def test_with_punctuation(self):
        assert accuracy("yes.", "yes") == 1.0


class TestRougeL:
    def test_exact_match(self):
        score = rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.99

    def test_no_overlap(self):
        score = rouge_l("hello", "goodbye")
        assert score == 0.0

    def test_partial(self):
        score = rouge_l("the cat sat on the mat", "the cat on the mat")
        assert 0.5 < score < 1.0


class TestLcsRougeL:
    """Test the fallback LCS implementation directly."""

    def test_exact_match(self):
        assert _lcs_rouge_l("hello world", "hello world") == 1.0

    def test_both_empty(self):
        assert _lcs_rouge_l("", "") == 1.0

    def test_pred_empty(self):
        assert _lcs_rouge_l("", "hello") == 0.0

    def test_gold_empty(self):
        assert _lcs_rouge_l("hello", "") == 0.0

    def test_partial(self):
        score = _lcs_rouge_l("a b c d", "a c d e")
        assert 0.0 < score < 1.0
