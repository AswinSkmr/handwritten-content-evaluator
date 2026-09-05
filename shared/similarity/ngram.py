"""
Traditional plagiarism detection - Method 2: N-gram overlap.

Unlike TF-IDF (which treats word order as irrelevant), n-gram overlap
compares sequences of N consecutive words. This is more sensitive to
phrase-level copying -- two documents sharing many identical trigrams
is much stronger evidence of direct copying than shared unigrams alone,
since it requires the same words in the same order.
"""


def get_ngrams(text: str, n: int) -> set[str]:
    """
    Returns the set of unique n-grams (as space-joined strings) from
    a piece of text. Uses a simple whitespace split -- text should
    already be normalized (see shared/preprocessing.py) before calling
    this, so we're not tripped up by inconsistent spacing/casing here.
    """
    words = text.split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def compute_ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """
    Returns the overlap ratio of n-grams between two texts: the size
    of the intersection divided by the size of the smaller n-gram set.

    Dividing by the smaller set (rather than the union, which is
    Jaccard's approach -- see jaccard.py) means a short text fully
    contained within a longer one scores close to 1.0, which is the
    behavior we want for plagiarism detection: a short copied sentence
    embedded in an otherwise original long document should still be
    flagged as highly overlapping, not diluted by the length difference.
    """
    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)

    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b
    smaller_set_size = min(len(ngrams_a), len(ngrams_b))

    return len(intersection) / smaller_set_size
