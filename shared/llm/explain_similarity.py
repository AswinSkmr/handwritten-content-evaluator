"""
M8 - LLM-assisted explanation of flagged similarity pairs.

Per the project's core principle, the LLM NEVER makes the plagiarism
decision itself. It is called only on pairs that deterministic
metrics (M6 lexical + M7 semantic) have already flagged as worth a
human's attention, and its sole job is to explain, in plain language,
WHY the pair looks similar -- summarizing evidence a reviewer would
otherwise have to piece together manually from raw scores.

Cost control (M18): this function is deliberately NOT called for
every pairwise comparison. At 1,770 pairs, calling an LLM on all of
them would be wasteful and slow. Callers should only invoke this for
pairs whose hybrid/lexical/semantic scores already exceed a
"requires review" threshold.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = None


def _get_client():
    """
    Lazily creates the Gemini client on first use, not at import time.
    This avoids requiring a valid API key just to import this module
    (e.g. during testing of unrelated code that imports shared.llm).
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Make sure it's set in your "
                ".env file at the project root."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def explain_similarity(
    text_a: str,
    text_b: str,
    lexical_score: float,
    semantic_score: float,
    matching_sentences: list[dict],
) -> str:
    """
    Generates a plain-language explanation of why two documents were
    flagged as similar, grounded in the actual scores and matching
    sentences already computed by M6/M7 -- the LLM is given the
    evidence, not asked to independently judge the documents.

    Returns the explanation text. Does NOT return a verdict, a
    percentage, or a plagiarism classification -- that remains the
    responsibility of the deterministic hybrid score (M14).
    """
    match_summary = "\n".join(
        f"- \"{m['sentence_a']}\" <-> \"{m['sentence_b']}\" "
        f"(similarity: {m['similarity']:.0%})"
        for m in matching_sentences[:5]  # cap at 5 to keep the prompt small
    )

    prompt = f"""You are assisting a plagiarism review process. You are
given similarity scores and matching sentence pairs that a deterministic
system already computed. Your job is ONLY to explain, in 2-3 sentences,
what kind of similarity this appears to be (e.g. direct copying, light
paraphrasing, coincidental shared terminology). Do NOT state a
percentage, a verdict, or claim to know for certain whether this is
plagiarism -- a human reviewer makes that judgment, not you.

Lexical similarity score: {lexical_score:.0%}
Semantic similarity score: {semantic_score:.0%}

Matching sentence pairs:
{match_summary if match_summary else "(no individual sentence matches above threshold)"}

Provide a brief, neutral explanation for the human reviewer."""

    client = _get_client()
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text