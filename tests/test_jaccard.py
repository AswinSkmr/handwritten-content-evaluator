from shared.similarity.jaccard import compute_jaccard_similarity

identical = "The quick brown fox jumps over the lazy dog."
similar = "A quick brown fox jumped over a lazy dog."
different = "Plagiarism detection systems use natural language processing."

short_copied_snippet = "the lazy dog"
long_doc_containing_snippet = (
    "This is a long original essay about animals in general. "
    "Somewhere in the middle, the lazy dog appears as a phrase. "
    "The rest of this document is completely unrelated content about history."
)

print(f"Identical texts (n=3):          {compute_jaccard_similarity(identical, identical, n=3):.4f}")
print(f"Similar texts (n=3):            {compute_jaccard_similarity(identical, similar, n=3):.4f}")
print(f"Different texts (n=3):          {compute_jaccard_similarity(identical, different, n=3):.4f}")
print(f"Short snippet in long doc (n=3): {compute_jaccard_similarity(short_copied_snippet, long_doc_containing_snippet, n=3):.4f}")