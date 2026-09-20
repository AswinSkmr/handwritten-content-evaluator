from shared.similarity.hybrid import compute_hybrid_score

test_cases = {
    "Identical text": (
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
    ),
    "HTR-noised exact copy (from M11)": (
        'assuredness " Bella Bella Marie " ( Parlophone ) a lively song that change\'s tempo midway .',
        'assuredness "Bella Bella Marie" (Parlophone), a lively song that changes tempo mid-way.',
    ),
    "Heavy paraphrase": (
        "The car was fast.",
        "The automobile was rapid.",
    ),
    "Unrelated": (
        "Plagiarism detection systems use natural language processing.",
        "The weather today is sunny with a chance of rain.",
    ),
}

for label, (text_a, text_b) in test_cases.items():
    result = compute_hybrid_score(text_a, text_b)
    print(f"--- {label} ---")
    print(f"  Lexical (avg): {result['lexical_combined']:.1%}  |  Semantic: {result['semantic']:.1%}")
    print(f"  Hybrid score:  {result['hybrid_score']:.1%}")
    print(f"  Category: {result['category']}\n")