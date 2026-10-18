"""
M11 - Handwritten-to-Plagiarism Pipeline: HTR extraction.

Takes a photo or scan of a handwritten page and returns plain text,
which then flows into the same shared preprocessing and similarity
code that Pipeline A uses.

Steps:
1. Load as grayscale.
2. Work out which way up the page is. Phone photos are often stored
   sideways, so two candidate rotations are tried (0/180 degrees if
   the text already runs horizontally, otherwise 90 degrees either
   way). TrOCR itself arbitrates: text read the right way up scores a
   much higher average log-probability than text read upside-down.
3. Optionally flatten uneven lighting.
4. Split the page into text lines (TrOCR reads one line at a time).
5. Recognise all lines in one batch and join them with newlines.

Pretrained TrOCR is the production model (fine-tuning was attempted
twice and did not beat the baseline; see the Model Information page).
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from pipeline_b_handwritten.preprocess_image import load_image, to_grayscale
from pipeline_b_handwritten.segment_lines import (
    horizontal_score, segment_lines, normalize_background, crop_lines,
)

_MODEL_NAME = "microsoft/trocr-large-handwritten"
_processor = None
_model = None
_device = None

_ROTATIONS = {
    "0": None,
    "180": cv2.ROTATE_180,
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
}


def _get_model():
    """Loads the pretrained TrOCR model on first use, not at import time."""
    global _processor, _model, _device
    if _model is None:
        _processor = TrOCRProcessor.from_pretrained(_MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(_MODEL_NAME)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()
    return _processor, _model, _device


def _recognise(crops: list):
    """
    Runs TrOCR on grayscale line crops in one batch. Returns the list
    of line texts and the mean beam-search score (average
    log-probability per token), used to compare orientations.

    Notes on generation parameters:
    - num_beams=5: wider beam search finds better transcriptions.
    - no_repeat_ngram_size is intentionally NOT set: in handwritten
      text, words and short phrases naturally repeat (e.g. "the the"
      across two lines). Blocking 3-gram repeats forces the model to
      produce wrong tokens, badly hurting accuracy and causing two
      copies of the same image to produce divergent transcripts.
    - length_penalty=1.0: neutral length bias (beam scores are
      already log-prob per token; this keeps them comparable across
      orientations without artificially preferring short sequences).
    """
    processor, model, device = _get_model()
    images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)) for c in crops]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        out = model.generate(
            pixel_values,
            num_beams=5,
            max_length=128,
            length_penalty=1.0,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    texts = [t.strip() for t in processor.batch_decode(out.sequences, skip_special_tokens=True)]
    scores = getattr(out, "sequences_scores", None)
    mean_score = float(scores.float().mean()) if scores is not None else 0.0
    return texts, mean_score


def _resize_for_trocr(gray: np.ndarray) -> np.ndarray:
    """
    TrOCR's processor internally resizes images, but very high-resolution
    photos (e.g. from phone cameras at 12MP+) cause two problems:
    1. Line segmentation becomes coarse (letter strokes are thin relative
       to the full image size), so segment_lines misses or merges lines.
    2. After TrOCR's internal downscale the characters become very small,
       degrading recognition accuracy.
    Cap the long edge at 2000 px while preserving aspect ratio, which
    keeps letter strokes at a size TrOCR was trained to handle.
    """
    h, w = gray.shape
    max_side = 2000
    if max(h, w) <= max_side:
        return gray
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    sharpen local ink contrast on a background-normalized page.

    Applied AFTER normalize_background (not before), so the paper
    background is already flat white and CLAHE only acts on the ink
    strokes — exactly where TrOCR needs more contrast. A gentle
    clipLimit=1.5 avoids amplifying scanner noise or JPEG artifacts.
    """
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _read_page(gray: np.ndarray, rotation: str, flatten: bool):
    rotated = gray if _ROTATIONS[rotation] is None else cv2.rotate(gray, _ROTATIONS[rotation])
    # Resize large photos before segmentation so line detection is accurate
    resized = _resize_for_trocr(rotated)
    prepared = normalize_background(resized) if flatten else resized
    # Sharpen local ink contrast after background is flattened
    prepared = _apply_clahe(prepared)

    boxes = segment_lines(prepared)
    # No lines found: treat the whole image as one line (pre-cropped
    # single-line images such as IAM end up here).
    crops = crop_lines(prepared, boxes) if boxes else [prepared]
    return _recognise(crops)


def extract_text_from_handwriting(file_path: str, flatten_background: bool = True) -> str:
    """
    Reads a handwritten image (JPG/PNG/scanned page) and returns its
    text, one recognised line per line.

    Raises:
        ValueError: if the image cannot be read, or no text is found.
    """
    gray = to_grayscale(load_image(file_path))

    candidates = ["0", "180"] if horizontal_score(gray) >= 1.0 else ["90ccw", "90cw"]

    best_texts, best_score = None, -np.inf
    for rotation in candidates:
        texts, score = _read_page(gray, rotation, flatten_background)
        if score > best_score:
            best_texts, best_score = texts, score

    text = "\n".join(t for t in best_texts if t)
    if not text.strip():
        raise ValueError(
            f"HTR produced no text for image: {file_path}. The image "
            "may be blank, illegible, or not contain handwriting."
        )
    return text