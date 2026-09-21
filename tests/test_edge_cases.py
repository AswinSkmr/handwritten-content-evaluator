"""
M17 - Edge case testing per the project's explicit requirements:
file validation, empty-input handling, corrupted/invalid files, and
duplicate-file handling.
"""

import pytest

from pipeline_a_digital.dispatcher import extract_text
from pipeline_a_digital.extract_txt import extract_text_from_txt
from pipeline_b_handwritten.preprocess_image import load_image


# --- Empty input handling ---

def test_empty_txt_file_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        extract_text_from_txt("data/test_samples/empty.txt")


# --- Unsupported / invalid file handling ---

def test_unsupported_extension_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported file extension"):
        extract_text("data/test_samples/sample1.jpg")


def test_nonexistent_file_raises_error():
    """
    A file path that doesn't exist at all should fail cleanly with
    some kind of error -- not silently return empty/wrong content.
    """
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        extract_text("data/test_samples/does_not_exist.txt")


def test_corrupted_image_raises_value_error(tmp_path):
    """
    A file with an image extension but non-image content (simulating
    a corrupted upload) should be caught, not crash unpredictably.
    """
    fake_image = tmp_path / "corrupted.png"
    fake_image.write_bytes(b"this is not actually image data")

    with pytest.raises(ValueError, match="Could not read image"):
        load_image(str(fake_image))


# --- Duplicate file handling ---

def test_comparing_identical_file_to_itself_scores_maximum():
    """
    'Duplicate file handling' interpreted as: comparing a document
    against an exact duplicate of itself should correctly max out
    similarity, not error out or behave unexpectedly.
    """
    from shared.similarity.hybrid import compute_hybrid_score

    text = extract_text("data/test_samples/sample1.txt")
    result = compute_hybrid_score(text, text)
    assert result["hybrid_score"] >= 0.99

# --- API failure handling ---

def test_explain_similarity_raises_clear_error_without_api_key(monkeypatch):
    """
    If GEMINI_API_KEY is missing or empty, this should fail with a
    clear, actionable error message -- not a cryptic library
    exception the user can't interpret during a live demo.
    """
    import shared.llm.explain_similarity as explain_module

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    explain_module._client = None  # reset the lazily-cached client

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY not found"):
        explain_module.explain_similarity(
            text_a="test", text_b="test",
            lexical_score=1.0, semantic_score=1.0,
            matching_sentences=[],
        )