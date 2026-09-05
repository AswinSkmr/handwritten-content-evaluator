"""
Traditional plagiarism detection - Method 4: Sentence-level matching.

Unlike the previous three methods, this doesn't return a single score
for a document pair -- it identifies WHICH specific sentence pairs are
similar, which is what M15's evidence reporting ("matching passages")
actually needs to show a human reviewer. A bare 91% doesn't help
anyone review a case; "these two sentences are near-identical" does.

Reuses the TF-IDF similarity function already built for sentence-pair
comparison, rather than introducing a new similarity primitive --
consistent with keeping shared logic in one place.
"""

from shared.preprocessing import segment_sentences
from shared.similarity.tfidf import compute_tfidf_similarity


def find_matching_sentences(
    text_a: str, text_b: str, threshold: float = 0.5
) -> list[dict]:
    """
    Compares every sentence in text_a against every sentence in text_b
    and returns pairs whose TF-IDF cosine similarity meets or exceeds
    the threshold.

    Returns a list of dicts, each with the two matched sentences, their
    similarity score, and their sentence index in each document (the
    index is useful later for highlighting matches in a UI).

    Note: this is O(n*m) in the number of sentences -- fine for
    individual document pairs, but a real concern at M13's 1,770-pair
    scale if documents are long. We'll revisit performance at M18.
    """
    sentences_a = segment_sentences(text_a)
    sentences_b = segment_sentences(text_b)

    matches = []
    for i, sentence_a in enumerate(sentences_a):
        for j, sentence_b in enumerate(sentences_b):
            score = compute_tfidf_similarity(sentence_a, sentence_b)
            if score >= threshold:
                matches.append({
                    "sentence_a_index": i,
                    "sentence_b_index": j,
                    "sentence_a": sentence_a,
                    "sentence_b": sentence_b,
                    "similarity": score,
                })

    return matches