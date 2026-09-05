"""
Pipeline A - Digital Documents
Extracts plain text from .pdf files using pdfplumber.

Handles:
1. Multi-page PDFs (concatenates text from all pages).
2. Scanned/image-only PDFs (no embedded text layer) -- these need OCR,
   which is out of scope here. We detect this case and raise a clear
   error rather than silently returning an empty string, since a silent
   empty result would look identical to a plagiarism score of "no
   similarity" instead of "we couldn't read this file."
3. Corrupted or unreadable PDF files.
4. Empty documents (same principle as extract_txt.py).
"""

import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """
    Reads a .pdf file and returns its concatenated text content.

    Raises:
        ValueError: if the PDF has no extractable text (likely scanned/
            image-only and needs OCR), or if all pages are empty.
        RuntimeError: if the PDF file is corrupted or cannot be opened.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            page_texts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
    except Exception as e:
        raise RuntimeError(f"Could not open or parse PDF file: {file_path} ({e})")

    full_text = "\n".join(page_texts)

    if not full_text.strip():
        raise ValueError(
            f"No extractable text found in PDF: {file_path}. "
            "This likely means it's a scanned/image-only PDF, which "
            "requires OCR (not supported by this text-extraction pipeline)."
        )

    return full_text