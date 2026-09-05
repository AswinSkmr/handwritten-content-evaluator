from shared.similarity.semantic import compute_semantic_similarity
from shared.similarity.tfidf import compute_tfidf_similarity

paraphrase_a = "The car was fast."
paraphrase_b = "The automobile was rapid."

identical = "The quick brown fox jumps over the lazy dog."
similar = "A quick brown fox jumped over a lazy dog."
different = "Plagiarism detection systems use natural language processing."

print("--- Paraphrase case (no shared vocabulary) ---")
print(f"Semantic similarity: {compute_semantic_similarity(paraphrase_a, paraphrase_b):.4f}")
print(f"TF-IDF similarity:   {compute_tfidf_similarity(paraphrase_a, paraphrase_b):.4f}")

print("\n--- Comparison on earlier M6 test cases ---")
print(f"Identical - Semantic: {compute_semantic_similarity(identical, identical):.4f}, TF-IDF: {compute_tfidf_similarity(identical, identical):.4f}")
print(f"Similar   - Semantic: {compute_semantic_similarity(identical, similar):.4f}, TF-IDF: {compute_tfidf_similarity(identical, similar):.4f}")
print(f"Different - Semantic: {compute_semantic_similarity(identical, different):.4f}, TF-IDF: {compute_tfidf_similarity(identical, different):.4f}")