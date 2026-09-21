"""
M17 - Proper pytest tests for the hybrid scorer, with real pass/fail
assertions instead of printed output a human has to eyeball.
"""

from shared.similarity.hybrid import compute_hybrid_score


def test_identical_text_scores_high(identical_text_pair):
    text_a, text_b = identical_text_pair
    result = compute_hybrid_score(text_a, text_b)
    assert result["hybrid_score"] >= 0.95
    assert result["category"] == "HIGH SIMILARITY - HUMAN REVIEW REQUIRED"


def test_unrelated_text_scores_low(unrelated_text_pair):
    text_a, text_b = unrelated_text_pair
    result = compute_hybrid_score(text_a, text_b)
    assert result["hybrid_score"] < 0.25
    assert result["category"] == "MINIMAL SIMILARITY - INDEPENDENT"


def test_paraphrase_scores_moderate_or_higher(paraphrase_text_pair):
    """
    A heavy paraphrase with zero shared vocabulary should still be
    caught by semantic similarity, landing at least in MODERATE --
    this is the whole point of M7's semantic layer existing.
    """
    text_a, text_b = paraphrase_text_pair
    result = compute_hybrid_score(text_a, text_b)
    assert result["semantic"] > 0.7
    assert result["hybrid_score"] >= 0.50


def test_hybrid_score_never_exceeds_reasonable_bounds(identical_text_pair):
    """
    Sanity check: the hybrid score should never wildly exceed 1.0,
    protecting against a future weighting-formula bug.
    """
    text_a, text_b = identical_text_pair
    result = compute_hybrid_score(text_a, text_b)
    assert -0.1 <= result["hybrid_score"] <= 1.1