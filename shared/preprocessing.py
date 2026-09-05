"""
Shared text preprocessing, used by both Pipeline A (digital documents)
and Pipeline B (handwritten documents, after HTR). This is the
convergence point described in M11 -- both pipelines produce raw
extracted text, and from here onward the plagiarism-detection logic
is identical regardless of source.

Design principle: preprocessing for plagiarism detection is NOT the
same as preprocessing for general NLP tasks (e.g. sentiment analysis).
The exact wording is evidence -- we deliberately avoid aggressive
transformations (stemming, stopword removal) that could destroy
signal needed to distinguish genuine paraphrasing from coincidental
similarity. See each function's docstring for the specific
justification of what it does and does not do.
"""

import re

import nltk


def normalize_whitespace(text: str) -> str:
    """
    Collapses runs of whitespace (multiple spaces, tabs, newlines) into
    single spaces, and strips leading/trailing whitespace.

    Safe transformation: no information loss relevant to plagiarism
    detection. Whitespace differences (e.g. one extra space) are not
    meaningful signal -- they're artifacts of formatting/copy-paste,
    not evidence of independent authorship.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_case(text: str, lowercase: bool = False) -> str:
    """
    Optionally lowercases text. OFF by default.

    Justification for defaulting to OFF: unusual or matching
    capitalization patterns across two documents (e.g. both
    incorrectly capitalizing a word mid-sentence) can itself be
    evidence of copying. Lowercasing by default would destroy that
    signal. We expose this as an explicit, opt-in choice rather than
    a hidden default, since M6's similarity methods may want to
    experiment with both settings and compare results.
    """
    return text.lower() if lowercase else text


def segment_sentences(text: str) -> list[str]:
    """
    Splits text into a list of sentences using NLTK's punkt tokenizer.

    This is needed for M6's sentence-level matching and M15's
    "matching passages" evidence -- both operate on sentences, not
    raw character offsets.
    """
    return nltk.sent_tokenize(text)


def preprocess(text: str, lowercase: bool = False) -> dict:
    """
    Runs the full preprocessing pipeline and returns both the
    normalized full text and its sentence segmentation, so downstream
    code has access to whichever granularity it needs.
    """
    normalized = normalize_whitespace(text)
    normalized = normalize_case(normalized, lowercase=lowercase)
    sentences = segment_sentences(normalized)

    return {
        "normalized_text": normalized,
        "sentences": sentences,
    }