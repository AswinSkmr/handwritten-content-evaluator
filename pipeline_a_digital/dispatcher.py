"""
Pipeline A - Digital Documents
Dispatches a file to the correct format-specific extractor based on
its file extension, and returns normalized (stripped) text.

This is the single entry point Pipeline A's downstream stages
(preprocessing, similarity detection) should import from -- they
should never need to know about extract_txt/extract_pdf/extract_docx
individually.
"""

import os

from pipeline_a_digital.extract_txt import extract_text_from_txt
from pipeline_a_digital.extract_pdf import extract_text_from_pdf
from pipeline_a_digital.extract_docx import extract_text_from_docx


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text(file_path: str) -> str:
    """
    Extracts and normalizes text from a supported digital document.

    Supported formats: .txt, .pdf, .docx

    Raises:
        ValueError: if the file extension is unsupported, or if the
            underlying extractor determines the file has no usable text
            (empty file, scanned/image-only PDF, etc.).
        RuntimeError: if the underlying extractor cannot open/parse the
            file at all (corrupted file).
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".txt":
        text = extract_text_from_txt(file_path)
    elif extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif extension == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{extension}' for file: {file_path}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Basic normalization at the dispatcher level: strip leading/trailing
    # whitespace. Deeper normalization (case, punctuation, sentence
    # segmentation) is M5's job, not this one -- extraction and
    # preprocessing are kept as separate, single-purpose stages.
    return text.strip()