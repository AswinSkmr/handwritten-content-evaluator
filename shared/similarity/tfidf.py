"""
Traditional plagiarism detection - Method 1: TF-IDF + cosine similarity.

Represents each document as a TF-IDF weighted vector, then measures
similarity as the cosine of the angle between two documents' vectors.
Common words get down-weighted automatically (they appear in most
documents, so their IDF weight is low); rare/distinctive shared
phrasing gets up-weighted.

This does NOT understand meaning -- "car" and "automobile" are treated
as completely unrelated. That's a known limitation, addressed later by
M7's embedding-based semantic similarity, not by this method.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarity(text_a: str, text_b: str) -> float:
    """
    Returns a similarity score between 0.0 (no lexical overlap) and
    1.0 (identical TF-IDF vectors) for two normalized text strings.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])

    # cosine_similarity returns a 2x2 matrix comparing both documents
    # to each other; [0][1] is the similarity of doc A to doc B.
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity_matrix[0][0])