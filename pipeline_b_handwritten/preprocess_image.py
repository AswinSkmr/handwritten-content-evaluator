"""
M9 - Handwritten image preprocessing.

Each transformation is implemented as its own function so we can test
them individually and in combination -- per the project's own
principle, we don't assume a step helps just because it's standard
practice. Thresholding/binarization in particular is applied only
optionally: TrOCR is a transformer trained on natural grayscale image
statistics, not a classical binarization-based OCR pipeline, so
aggressive binarization could hurt rather than help. This needs to be
tested empirically (see tests/test_preprocess_image.py), not assumed.
"""

import cv2
import numpy as np


def load_image(file_path: str) -> np.ndarray:
    """
    Loads an image from disk as a BGR numpy array (OpenCV's default).

    Raises:
        ValueError: if the file cannot be read as an image (corrupted,
            unsupported format, or path doesn't exist).
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image file: {file_path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to single-channel grayscale. Removes color
    information that's irrelevant to handwriting shape recognition,
    and is the expected input format for most downstream steps here.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray_image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve contrast, particularly useful for photos taken under
    uneven lighting (e.g. a shadow falling across part of a page) --
    a real concern for phone-photographed submissions, unlike IAM's
    uniformly-scanned samples.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def denoise(gray_image: np.ndarray) -> np.ndarray:
    """
    Applies non-local means denoising to reduce phone-camera sensor
    noise, which is more prominent in photographed pages than in
    clean scanner output.
    """
    return cv2.fastNlMeansDenoising(gray_image, h=10)


def deskew(gray_image: np.ndarray) -> np.ndarray:
    """
    Detects and corrects slight rotation (skew) in the image, common
    when a page is photographed at a slight angle rather than
    perfectly flat under a scanner. Uses the minimum-area bounding
    rectangle of dark (text) pixels to estimate skew angle.
    """
    coords = np.column_stack(np.where(gray_image < 128))
    if len(coords) == 0:
        return gray_image  # no dark pixels found, nothing to deskew

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns an angle in a somewhat unintuitive range;
    # this normalizes it to a small rotation correction rather than a
    # full 90-degree reorientation.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray_image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray_image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def binarize(gray_image: np.ndarray) -> np.ndarray:
    """
    Converts to pure black/white using Otsu's automatic thresholding.
    Kept SEPARATE and optional (not part of a default pipeline) --
    see this module's docstring for why this needs empirical testing
    against TrOCR specifically, rather than being assumed beneficial.
    """
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def get_default_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Applies the preprocessing pipeline actually justified by testing
    (see tests/test_preprocess_image.py and project documentation for
    the full evidence): grayscale conversion + deskew only.

    Contrast enhancement, denoising, and binarization were each tested
    independently and found to HURT accuracy on both a clean IAM
    sample and a synthetically degraded (rotated/uneven-lit/noisy)
    version of it -- likely because TrOCR, as a transformer trained on
    natural image statistics, is sensitive to the artifacts these
    classical techniques introduce. They remain available as separate
    functions in this module for experimentation, but are deliberately
    excluded from the default pipeline based on this evidence.

    Note: each step was tested independently, not in combination, on
    the degraded test case. Whether a combined pipeline (e.g. deskew +
    contrast together) performs better than either alone is untested
    and documented as a limitation / future work item, not assumed.
    """
    gray = to_grayscale(image)
    return deskew(gray)