"""
Synthetically degrades a clean IAM sample image to approximate the
kinds of problems a real phone-photographed page has, which clean
scanner images don't: slight rotation, uneven lighting (a soft
shadow gradient), and sensor noise. Used only for testing M9
preprocessing steps against a "messy" case until real photographed
samples are available.
"""

import cv2
import numpy as np

image = cv2.imread("data/test_samples/iam_sample.png")
h, w = image.shape[:2]

# 1. Slight rotation (simulating a page photographed at a small angle)
rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 4, 1.0)
rotated = cv2.warpAffine(
    image, rotation_matrix, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
)

# 2. Uneven lighting: a soft brightness gradient across the image,
# simulating a shadow falling across part of a photographed page.
gradient = np.tile(np.linspace(0.6, 1.3, w), (h, 1)).astype(np.float32)
gradient = cv2.merge([gradient, gradient, gradient])
lit = np.clip(rotated.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

# 3. Sensor noise (Gaussian noise, simulating phone camera sensor grain)
noise = np.random.normal(0, 15, lit.shape).astype(np.float32)
noisy = np.clip(lit.astype(np.float32) + noise, 0, 255).astype(np.uint8)

cv2.imwrite("data/test_samples/iam_sample_degraded.png", noisy)
print("Saved degraded image to data/test_samples/iam_sample_degraded.png")