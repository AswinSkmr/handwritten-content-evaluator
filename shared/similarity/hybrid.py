"""
M14 - Hybrid plagiarism score.

Combines M6's lexical methods (TF-IDF, n-gram overlap, Jaccard) and
M7's semantic similarity into one score, with a category label rather
than a bare percentage. Explicitly does NOT implement a naive
"similarity > X% = plagiarism" rule -- see the project's core
plagiarism principle (never claim to prove academic guilt; output
reads as "requires human review").

IMPORTANT LIMITATION, stated honestly: the weights below are a
REASONED STARTING POINT grounded in evidence already gathered during
M6/M7/M11 testing (see comments below), NOT weights tuned against a
labeled validation dataset. M12's controlled dataset (pending) is
needed to properly validate or adjust these weights and thresholds.
This is documented here rather than silently presented as final.

KNOWN LIMITATION (n=1 evidence): testing this scorer on the one real
cross-pipeline case available (M11's HTR-noised exact copy) produced
a hybrid score of 72% ("MODERATE" category), just under the 75% HIGH
threshold -- despite the underlying text being an exact copy. This
happened because n-gram overlap and Jaccard (both strict, word-order-
sensitive methods) dropped sharply under minor HTR noise, pulling the
3-method lexical average down to 32.6% even though TF-IDF alone and
semantic similarity both remained high. This was NOT "fixed" by
adjusting weights or thresholds against this single example, since
doing so would mean tuning on n=1 -- a scientifically weaker choice
than acknowledging the limitation and deferring real threshold
validation to M12's labeled dataset.
"""
from shared.similarity.tfidf import compute_tfidf_similarity
from shared.similarity.ngram import compute_ngram_overlap
from shared.similarity.jaccard import compute_jaccard_similarity
from shared.similarity.semantic import compute_semantic_similarity
from shared.similarity.sentence_matching import find_matching_sentences

# Semantic weighted higher than lexical: M11's testing showed semantic
# similarity stayed robust to HTR noise (98.2%) while lexical methods
# were more sensitive to surface-level differences (69.1% on the same
# genuinely-matching pair). Weights are NOT yet tuned against labeled
# data -- see module docstring.
LEXICAL_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

# Category thresholds on the final hybrid score. Also a reasoned
# starting point, not validated against labeled data yet.
THRESHOLD_HIGH = 0.75
THRESHOLD_MODERATE = 0.50
THRESHOLD_LOW = 0.25


def compute_hybrid_score(text_a: str, text_b: str) -> dict:
    """
    Computes a hybrid similarity score and category for two
    (already-preprocessed, normalized) text strings.

    Returns a dict with the individual method scores, the combined
    lexical sub-score, the semantic score, the final hybrid score, a
    category label, and matching sentence evidence -- never just a
    bare number. Per the project's core principle, the category is
    phrased as a review recommendation, not a plagiarism verdict.
    """
    tfidf = compute_tfidf_similarity(text_a, text_b)
    ngram = compute_ngram_overlap(text_a, text_b)
    jaccard = compute_jaccard_similarity(text_a, text_b)
    semantic = compute_semantic_similarity(text_a, text_b)

    lexical_combined = (tfidf + ngram + jaccard) / 3

    hybrid_score = (LEXICAL_WEIGHT * lexical_combined) + (SEMANTIC_WEIGHT * semantic)

    if hybrid_score >= THRESHOLD_HIGH:
        category = "HIGH SIMILARITY - HUMAN REVIEW REQUIRED"
    elif hybrid_score >= THRESHOLD_MODERATE:
        category = "MODERATE SIMILARITY - REVIEW RECOMMENDED"
    elif hybrid_score >= THRESHOLD_LOW:
        category = "LOW SIMILARITY - LIKELY INDEPENDENT"
    else:
        category = "MINIMAL SIMILARITY - INDEPENDENT"

    matches = find_matching_sentences(text_a, text_b, threshold=0.5)

    return {
        "tfidf": tfidf,
        "ngram_overlap": ngram,
        "jaccard": jaccard,
        "lexical_combined": lexical_combined,
        "semantic": semantic,
        "hybrid_score": hybrid_score,
        "category": category,
        "matching_sentences": matches,
    }