from shared.llm.explain_similarity import explain_similarity

text_a = "Machine learning is a subset of artificial intelligence."
text_b = "Machine learning is a subset of artificial intelligence."

matches = [
    {
        "sentence_a": "Machine learning is a subset of artificial intelligence.",
        "sentence_b": "Machine learning is a subset of artificial intelligence.",
        "similarity": 1.0,
    }
]

explanation = explain_similarity(
    text_a=text_a,
    text_b=text_b,
    lexical_score=1.0,
    semantic_score=1.0,
    matching_sentences=matches,
)

print("Explanation:")
print(explanation)