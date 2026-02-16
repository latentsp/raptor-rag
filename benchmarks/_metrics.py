"""
Pure metric functions for RAPTOR benchmarks.

All functions operate on strings and return floats in [0, 1].
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles/punctuation/whitespace. SQuAD-style normalization."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token-level F1 between two strings."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(prediction: str, golds: list[str]) -> float:
    """Max F1 across multiple gold answers."""
    if not golds:
        return 0.0
    return max(token_f1(prediction, g) for g in golds)


def accuracy(predicted: str, correct: str) -> float:
    """Normalized exact-match accuracy (0.0 or 1.0)."""
    return 1.0 if normalize_answer(predicted) == normalize_answer(correct) else 0.0


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 score. Uses rouge-score lib if available, falls back to LCS."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(gold, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        return _lcs_rouge_l(prediction, gold)


def _lcs_rouge_l(prediction: str, gold: str) -> float:
    """Fallback ROUGE-L using longest common subsequence."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def meteor(prediction: str, gold: str) -> float:
    """METEOR score using nltk. Returns 0.0 if nltk unavailable."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score as _meteor

        # Ensure wordnet data is available
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)

        pred_tokens = prediction.split()
        gold_tokens = gold.split()
        if not gold_tokens:
            return 1.0 if not pred_tokens else 0.0
        return _meteor([gold_tokens], pred_tokens)
    except ImportError:
        return 0.0
