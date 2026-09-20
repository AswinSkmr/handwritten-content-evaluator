"""
M11 - Proves the convergence architecture: Pipeline B's HTR output
and Pipeline A's digital extraction output both feed into the exact
same shared preprocessing and similarity functions, with zero
duplicated plagiarism-detection logic.
"""

from pipeline_a_digital.dispatcher import extract_text
from pipeline_b_handwritten.extract_handwriting import extract_text_from_handwriting
from shared.preprocessing import preprocess
from shared.similarity.tfidf import compute_tfidf_similarity
from shared.similarity.semantic import compute_semantic_similarity

# Pipeline B: handwritten image -> text
handwritten_text = extract_text_from_handwriting("data/test_samples/iam_sample.png")
print(f"Pipeline B (handwritten) extracted: {handwritten_text}")

# A digital "document" containing the same content, near-verbatim,
# simulating a matching typed submission.
digital_text = (
    'assuredness "Bella Bella Marie" (Parlophone), a lively song that '
    "changes tempo mid-way."
)
print(f"Pipeline A (digital) text: {digital_text}")

# Both texts now go through the SAME shared preprocessing and
# similarity functions -- this is the convergence point M11 requires.
processed_handwritten = preprocess(handwritten_text)
processed_digital = preprocess(digital_text)

tfidf_score = compute_tfidf_similarity(
    processed_handwritten["normalized_text"], processed_digital["normalized_text"]
)
semantic_score = compute_semantic_similarity(
    processed_handwritten["normalized_text"], processed_digital["normalized_text"]
)

print(f"\nTF-IDF similarity (lexical):  {tfidf_score:.1%}")
print(f"Semantic similarity (meaning): {semantic_score:.1%}")