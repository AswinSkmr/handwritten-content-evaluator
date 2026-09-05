"""
Pipeline A - Digital Documents
Extracts plain text from .txt files.

Handles two real-world issues:
1. Encoding: not all .txt files are UTF-8 (Windows-saved files are
   often cp1252/latin-1). We try UTF-8 first, then fall back.
2. Empty files: an empty submission should be flagged, not silently
   treated as valid "empty" text that compares as 0% similar to everything.
"""


def extract_text_from_txt(file_path: str) -> str:
    """
    Reads a .txt file and returns its raw text content.

    Raises:
        ValueError: if the file is empty or contains only whitespace.
    """
    text = None

    # Try UTF-8 first (the common case), fall back to latin-1 if that fails.
    # latin-1 can decode ANY byte sequence without erroring, so it's a safe
    # last resort -- but it's tried second because it can silently produce
    # wrong characters for text that was actually UTF-8 with an encoding
    # bug elsewhere.
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError(f"Could not decode file with utf-8 or latin-1: {file_path}")

    if not text.strip():
        raise ValueError(f"File is empty or contains only whitespace: {file_path}")

    return text