"""
M7 - Semantic similarity using pretrained sentence embeddings.

Unlike every method in M6 (TF-IDF, n-gram overlap, Jaccard, sentence
matching), which all operate on exact word overlap, this method
captures MEANING. "The car was fast" and "The automobile was rapid"
score near-zero on every M6 method (no shared vocabulary) but should
score highly here, since they mean the same thing. This directly
addresses the "heavy paraphrasing" category from the project's core
plagiarism principle, which lexical methods alone cannot detect.

Model: all-MiniLM-L6-v2 -- a small (~80MB), fast, well-established
sentence embedding model. Chosen over a larger LLM-based embedding
model for a deliberate speed/resource/accuracy trade-off appropriate
for this project's scale (see project documentation for the full
justification of this choice, matching the 5 explanation points
required for the LLM/embedding model selection).
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Loaded once at module level, not inside the function -- loading the
# model from disk takes real time (a few hundred ms to seconds), so
# repeatedly reloading it for every comparison would be wasteful,
# especially at M13's 1,770-pair scale. This is also directly relevant
# to M18's performance optimization concerns.
_model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Returns a semantic similarity score between 0.0 and 1.0 (in
    practice, cosine similarity of sentence embeddings usually falls
    between roughly 0.0 and 1.0 for related text, but can technically
    go slightly negative for very unrelated text).
    """
    embeddings = _model.encode([text_a, text_b])
    similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    return float(similarity_matrix[0][0])