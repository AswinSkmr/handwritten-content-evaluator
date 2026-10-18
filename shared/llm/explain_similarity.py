"""
Explainable similarity intelligence: generates plain-language forensic
explanations for flagged document pairs based on lexical and semantic evidence.
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

_client = None
_cached_key = None


def _get_client(api_key: str = None):
    """
    Lazily creates the Gemini client on first use or when key changes.
    """
    global _client, _cached_key
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_key:
        return None
    if _client is None or effective_key != _cached_key:
        try:
            from google import genai
            _client = genai.Client(api_key=effective_key)
            _cached_key = effective_key
        except Exception:
            return None
    return _client


def generate_deterministic_explanation(
    lexical_score: float,
    semantic_score: float,
    matching_sentences: list[dict],
) -> str:
    """
    High-accuracy deterministic forensic explanation when LLM is unavailable or offline.
    Evaluates lexical vs semantic divergences, match count, and paraphrase signatures.
    """
    num_matches = len(matching_sentences)
    highest_match = matching_sentences[0]["similarity"] if matching_sentences else 0.0

    # Case 1: High lexical + High semantic (Verbatim copying / near duplicate)
    if lexical_score >= 0.70 and semantic_score >= 0.75:
        return (
            f"The documents exhibit strong evidence of verbatim or near-verbatim text duplication "
            f"({lexical_score:.0%} lexical overlap and {semantic_score:.0%} semantic congruence). "
            f"{num_matches} matching sentence passage(s) were identified with peak sentence alignment "
            f"at {highest_match:.0%}, indicating substantial shared phrasing rather than independent derivation."
        )

    # Case 2: Low lexical + High semantic (Paraphrasing / concept rewriting)
    if lexical_score < 0.40 and semantic_score >= 0.65:
        return (
            f"The documents exhibit conceptual alignment with significant vocabulary substitution "
            f"({semantic_score:.0%} semantic similarity against only {lexical_score:.0%} lexical overlap). "
            f"This pattern strongly suggests intentional paraphrasing or restructured presentation of the "
            f"same underlying ideas rather than direct cut-and-paste copying."
        )

    # Case 3: Moderate lexical + Moderate semantic (Shared topic / standard domain terminology)
    if 0.35 <= lexical_score < 0.70 and 0.40 <= semantic_score < 0.75:
        return (
            f"The comparison reveals moderate overlap ({lexical_score:.0%} lexical, {semantic_score:.0%} semantic). "
            f"This is typical of documents addressing the same academic prompt or domain-specific topic, "
            f"sharing technical vocabulary and formulas. While {num_matches} sentence pair(s) aligned, "
            f"human inspection is recommended to verify whether the shared content reflects standard definitions or copied structure."
        )

    # Case 4: Minimal or Low similarity
    if lexical_score < 0.30 and semantic_score < 0.45:
        return (
            f"The documents appear largely independent ({lexical_score:.0%} lexical, {semantic_score:.0%} semantic). "
            f"No extended continuous passages exceed suspicion thresholds, and the observed similarity is consistent "
            f"with random coincidence or basic common language."
        )

    # General catch-all
    return (
        f"The comparison indicates an overall hybrid similarity driven by {lexical_score:.0%} lexical overlap "
        f"and {semantic_score:.0%} semantic similarity across {num_matches} flagged sentence pair(s). "
        f"A human reviewer should inspect the highlighted passages below to evaluate academic integrity."
    )


def explain_similarity(
    text_a: str,
    text_b: str,
    lexical_score: float,
    semantic_score: float,
    matching_sentences: list[dict],
    api_key: str = None,
) -> str:
    """
    Generates a plain-language explanation of why two documents were
    flagged as similar, grounded in the actual scores and matching
    sentences already computed by M6/M7.

    Raises RuntimeError if GEMINI_API_KEY is not found.
    """
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Make sure it's set in your "
            ".env file at the project root."
        )

    res = explain_similarity_detailed(
        text_a=text_a,
        text_b=text_b,
        lexical_score=lexical_score,
        semantic_score=semantic_score,
        matching_sentences=matching_sentences,
        api_key=effective_key,
    )
    return res["text"]


def explain_similarity_detailed(
    text_a: str,
    text_b: str,
    lexical_score: float,
    semantic_score: float,
    matching_sentences: list[dict],
    api_key: str = None,
) -> dict:
    """
    Returns a dict with 'text', 'source' ('Google Gemini' or 'Deterministic NLP Engine'),
    and 'status' ('success' or 'fallback').
    """
    match_summary = "\n".join(
        f"- \"{m['sentence_a']}\" <-> \"{m['sentence_b']}\" "
        f"(similarity: {m['similarity']:.0%})"
        for m in matching_sentences[:5]
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

    client = _get_client(api_key=api_key)
    if client:
        # Try candidate modern models
        for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and hasattr(response, "text") and response.text:
                    return {
                        "text": response.text.strip(),
                        "source": f"Google Gemini ({model_name})",
                        "status": "success",
                    }
            except Exception:
                continue

    # Fallback to deterministic NLP engine
    fallback_text = generate_deterministic_explanation(
        lexical_score=lexical_score,
        semantic_score=semantic_score,
        matching_sentences=matching_sentences,
    )
    return {
        "text": fallback_text,
        "source": "Deterministic NLP Forensic Engine",
        "status": "fallback",
    }