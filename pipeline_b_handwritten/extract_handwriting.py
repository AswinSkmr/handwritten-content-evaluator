"""
M11 - Handwritten-to-Plagiarism Pipeline: HTR extraction.

Connects M9's image preprocessing to the pretrained TrOCR model (our
production HTR component, per the M10 decision -- fine-tuning was
attempted twice and documented as a negative result; the pretrained
baseline, at 5.56% CER on held-out IAM data, is what production uses).

This module's output (a plain text string) is deliberately the same
shape as pipeline_a_digital.dispatcher.extract_text()'s output --
both feed into the exact same shared/preprocessing.py and
shared/similarity/* functions. This is the convergence point M11
requires: no plagiarism-detection logic is duplicated between the
two pipelines, only the path getting TO plain text differs.
"""

import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from pipeline_b_handwritten.preprocess_image import load_image, get_default_pipeline

_MODEL_NAME = "microsoft/trocr-base-handwritten"
_processor = None
_model = None
_device = None


def _get_model():
    """
    Lazily loads the pretrained TrOCR model on first use, not at
    import time. Loading a ~1.3GB model just because this module was
    imported (e.g. by test code exploring unrelated parts of the
    project) would be wasteful -- same lazy-loading principle used in
    shared/llm/explain_similarity.py for the same reason.
    """
    global _processor, _model, _device
    if _model is None:
        _processor = TrOCRProcessor.from_pretrained(_MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(_MODEL_NAME)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()
    return _processor, _model, _device


def extract_text_from_handwriting(file_path: str) -> str:
    """
    Reads a handwritten image (JPG/PNG/scanned page), applies M9's
    evidence-based default preprocessing (grayscale + deskew), and
    runs it through the pretrained TrOCR model to produce plain text.

    Uses beam search (num_beams=4) and no_repeat_ngram_size=3 during
    generation -- these were found necessary during M10's diagnostic
    work to avoid repetition-loop artifacts in greedy decoding.

    Raises:
        ValueError: if the image file cannot be read (see
            preprocess_image.load_image).
    """
    processor, model, device = _get_model()

    raw_image = load_image(file_path)
    processed_image = get_default_pipeline(raw_image)

    # TrOCR expects RGB; our preprocessed image is single-channel
    # grayscale at this point.
    rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

    pixel_values = processor(images=rgb_image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values, num_beams=4, no_repeat_ngram_size=3, max_length=96
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if not text.strip():
        raise ValueError(
            f"HTR produced no text for image: {file_path}. The image "
            "may be blank, illegible, or not contain handwriting."
        )

    return text.strip()
    