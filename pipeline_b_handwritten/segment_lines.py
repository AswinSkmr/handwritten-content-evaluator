"""
M9/M11 - Page-level preparation for handwriting recognition.

TrOCR reads ONE line of text at a time (it was trained on single-line
images). A phone photo of a handwritten page therefore has to be split
into lines before recognition, and it must be the right way up first.
The earlier M9 experiments used IAM images, which are already single
upright lines, so these two steps were never needed until real photos
were used.

Everything here is plain OpenCV, with no model involved.
"""

import cv2
import numpy as np


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Binary mask where ink is white. An adaptive threshold is used
    because phone photos have uneven lighting; a single global
    threshold (like the one in preprocess_image.deskew) treats shadows
    and mid-grey paper as ink.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15
    )


def horizontal_score(gray: np.ndarray) -> float:
    """
    Above 1.0 means text lines run horizontally; below 1.0 means they
    run vertically (the photo is sideways).

    Horizontal text makes the per-row ink totals swing between "line"
    and "gap" rows, so their spread is larger than the spread of the
    per-column totals. Raw totals are used on purpose: a version
    normalised by image size misjudged tight single-line crops.
    """
    ink = (ink_mask(gray) > 0).astype(np.float32)
    return float(ink.sum(axis=1).std() / (ink.sum(axis=0).std() + 1e-6))


def _merge_same_line(boxes, min_overlap=0.3):
    """Union boxes whose vertical extents overlap enough to be one text line."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            _, yi, _, hi = boxes[i]
            _, yj, _, hj = boxes[j]
            overlap = min(yi + hi, yj + hj) - max(yi, yj)
            if overlap > 0 and overlap / min(hi, hj) >= min_overlap:
                parent[find(i)] = find(j)

    groups = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)

    merged = []
    for members in groups.values():
        x0 = min(b[0] for b in members)
        y0 = min(b[1] for b in members)
        x1 = max(b[0] + b[2] for b in members)
        y1 = max(b[1] + b[3] for b in members)
        merged.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    merged.sort(key=lambda b: b[1])
    return merged


def segment_lines(gray: np.ndarray) -> list:
    """
    Finds text lines in an upright page. Returns (x, y, w, h) boxes,
    top to bottom. Words are first joined horizontally into blobs, then
    blobs that share a horizontal band are merged into one line.

    Returns an empty list if nothing line-like is found; callers
    should then treat the whole image as a single line (this is the
    case for pre-cropped single-line images such as IAM).
    """
    h_img, w_img = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w_img // 18), 5))
    joined = cv2.morphologyEx(ink_mask(gray), cv2.MORPH_CLOSE, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        # Skip specks, and skip page edges / shadows (too tall to be a line).
        if w < w_img * 0.04 or h < h_img * 0.02 or h > h_img * 0.5:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return _merge_same_line(boxes)


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """
    Flattens uneven lighting: estimates the paper background by erasing
    the thin ink strokes, then divides the image by it. The result is
    dark ink on an even white page, closer to the clean scans TrOCR
    was trained on. Whether this actually helps recognition is tested
    in tests/check_real_photos.py rather than assumed.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 21)
    return cv2.divide(gray, background, scale=255)


def crop_lines(gray: np.ndarray, boxes: list, pad_ratio: float = 0.12) -> list:
    """Cuts each line out with a little padding (relative to line height)."""
    h_img, w_img = gray.shape
    crops = []
    for (x, y, w, h) in boxes:
        pad = int(h * pad_ratio)
        crops.append(
            gray[max(0, y - pad): min(h_img, y + h + pad),
                 max(0, x - pad): min(w_img, x + w + pad)]
        )
    return crops