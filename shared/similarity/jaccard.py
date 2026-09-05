"""
Traditional plagiarism detection - Method 3: Jaccard similarity.

Unlike ngram.py's overlap coefficient (which divides by the SMALLER
set, favoring detection of a short copied snippet inside a longer
document), Jaccard divides by the UNION of both sets -- meaning a
large length mismatch between two documents actively lowers the
score. This makes Jaccard better suited for comparing two documents
of roughly similar length/structure (e.g. two full essay submissions),
where a big length difference is itself evidence the documents aren't
simply copies of each other.
"""

from shared.similarity.ngram import get_ngrams


def compute_jaccard_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """
    Returns the Jaccard similarity of n-grams between two texts: the
    size of the intersection divided by the size of the union.

    Reuses get_ngrams from ngram.py rather than reimplementing n-gram
    extraction -- both methods share the same notion of "what counts
    as an n-gram," they just differ in how they turn set overlap into
    a score.
    """
    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)

    if not ngrams_a and not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b

    return len(intersection) / len(union) if union else 0.0