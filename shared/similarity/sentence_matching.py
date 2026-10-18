"""
Traditional plagiarism detection - Method 4: Sentence-level matching.

Unlike the previous three methods, this doesn't return a single score
for a document pair -- it identifies WHICH specific sentence pairs are
similar, which is what M15's evidence reporting ("matching passages")
actually needs to show a human reviewer. A bare 91% doesn't help
anyone review a case; "these two sentences are near-identical" does.

Upgraded to use a hybrid sentence-level score that combines:
  - TF-IDF cosine similarity (lexical / word-overlap evidence)
  - Semantic cosine similarity (meaning-level / paraphrase evidence)

Using TF-IDF alone meant that paraphrased sentences (same meaning,
different words) were missed. Semantic alone means typo-ridden HTR
output of the *same* sentence sometimes scores low because the garbled
token embeddings differ. The hybrid catches both cases: if EITHER the
words OR the meaning match, the pair surfaces as a candidate.
"""

from shared.preprocessing import segment_sentences
from shared.similarity.tfidf import compute_tfidf_similarity
from shared.similarity.semantic import compute_semantic_similarity

# Weight semantic slightly higher than lexical at sentence level.
# At document level M14 already does 40/60 lexical/semantic weighting;
# here we match that philosophy because paraphrase detection matters
# most at the sentence grain.
_SENT_LEXICAL_W = 0.4
_SENT_SEMANTIC_W = 0.6


def sentence_hybrid_score(sentence_a: str, sentence_b: str) -> float:
    """
    Returns a combined lexical + semantic similarity for a single
    sentence pair.  Falls back gracefully: if either scorer returns
    a non-finite value the other scorer's result is used alone.
    """
    lexical = compute_tfidf_similarity(sentence_a, sentence_b)
    semantic = compute_semantic_similarity(sentence_a, sentence_b)
    # Clamp semantic to [0, 1] — cosine can be slightly negative for very
    # unrelated text, which would unfairly pull the combined score down.
    semantic = max(0.0, semantic)
    return _SENT_LEXICAL_W * lexical + _SENT_SEMANTIC_W * semantic


def find_matching_sentences(
    text_a: str, text_b: str, threshold: float = 0.5
) -> list[dict]:
    """
    Compares every sentence in text_a against every sentence in text_b
    and returns pairs whose hybrid (lexical + semantic) similarity
    meets or exceeds the threshold.

    Returns a list of dicts sorted by descending similarity, each with:
    - sentence_a / sentence_b: the raw sentence strings
    - sentence_a_index / sentence_b_index: position in each document
      (useful for highlighting matches in the UI)
    - similarity: the combined hybrid score
    - lexical: the TF-IDF component alone (for display / breakdown)
    - semantic: the semantic component alone (for display / breakdown)

    Note: this is O(n*m) in the number of sentences -- fine for
    individual document pairs, but a real concern at M13's 1,770-pair
    scale if documents are long. We'll revisit performance at M18.
    """
    sentences_a = segment_sentences(text_a)
    sentences_b = segment_sentences(text_b)

    matches = []
    for i, sentence_a in enumerate(sentences_a):
        for j, sentence_b in enumerate(sentences_b):
            lexical = compute_tfidf_similarity(sentence_a, sentence_b)
            semantic = max(0.0, compute_semantic_similarity(sentence_a, sentence_b))
            score = _SENT_LEXICAL_W * lexical + _SENT_SEMANTIC_W * semantic
            if score >= threshold:
                matches.append({
                    "sentence_a_index": i,
                    "sentence_b_index": j,
                    "sentence_a": sentence_a,
                    "sentence_b": sentence_b,
                    "similarity": score,
                    "lexical": lexical,
                    "semantic": semantic,
                })

    # Sort best matches first so the UI shows the strongest evidence at top
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return matches