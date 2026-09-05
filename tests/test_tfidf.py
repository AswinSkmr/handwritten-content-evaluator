from shared.similarity.tfidf import compute_tfidf_similarity

identical = "The quick brown fox jumps over the lazy dog."
similar = "A quick brown fox jumped over a lazy dog."
different = "Plagiarism detection systems use natural language processing."

print(f"Identical texts: {compute_tfidf_similarity(identical, identical):.4f}")
print(f"Similar texts:   {compute_tfidf_similarity(identical, similar):.4f}")
print(f"Different texts: {compute_tfidf_similarity(identical, different):.4f}")