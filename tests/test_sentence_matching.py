from shared.similarity.sentence_matching import find_matching_sentences

doc_a = (
    "Machine learning is a subset of artificial intelligence. "
    "It allows systems to learn from data automatically. "
    "The weather today is sunny and warm."
)

doc_b = (
    "The weather today is sunny and warm. "
    "Deep learning is a specialized area of computer science. "
    "Machine learning is a subset of artificial intelligence."
)

matches = find_matching_sentences(doc_a, doc_b, threshold=0.5)

print(f"Found {len(matches)} matching sentence pair(s):\n")
for m in matches:
    print(f"  A[{m['sentence_a_index']}]: {m['sentence_a']}")
    print(f"  B[{m['sentence_b_index']}]: {m['sentence_b']}")
    print(f"  Similarity: {m['similarity']:.4f}\n")