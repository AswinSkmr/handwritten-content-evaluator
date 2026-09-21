"""
Shared pytest fixtures. pytest automatically discovers this file and
makes its fixtures available to every test in this folder, without
needing to import them manually.
"""

import pytest


@pytest.fixture
def identical_text_pair():
    text = "The quick brown fox jumps over the lazy dog."
    return text, text


@pytest.fixture
def unrelated_text_pair():
    return (
        "Plagiarism detection systems use natural language processing.",
        "The weather today is sunny with a chance of rain.",
    )


@pytest.fixture
def paraphrase_text_pair():
    return ("The car was fast.", "The automobile was rapid.")