import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer
import cv2

from pipeline_b_handwritten.preprocess_image import (
    load_image, to_grayscale, enhance_contrast, denoise, deskew, binarize,
    get_default_pipeline,
)

TRUE_TEXT = 'assuredness " Bella Bella Marie " ( Parlophone ) , a lively song that changes tempo mid-way .'

model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def predict(image_array):
    # TrOCR expects RGB; OpenCV images are single or triple-channel
    # numpy arrays -- convert grayscale back to 3-channel RGB if needed.
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    else:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    pixel_values = processor(images=image_array, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, num_beams=4, no_repeat_ngram_size=3, max_length=96)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def report(label, prediction):
    error = cer(TRUE_TEXT, prediction)
    print(f"{label}:")
    print(f"  Predicted: {prediction}")
    print(f"  CER: {error:.4f}\n")


raw = load_image("data/test_samples/iam_sample.png")
gray = to_grayscale(raw)
contrast = enhance_contrast(gray)
denoised = denoise(gray)
deskewed = deskew(gray)
binary = binarize(gray)

print(f"Ground truth: {TRUE_TEXT}\n")
report("Raw (no preprocessing)", predict(raw))
report("Grayscale only", predict(gray))
report("Grayscale + contrast enhancement", predict(contrast))
report("Grayscale + denoise", predict(denoised))
report("Grayscale + deskew", predict(deskewed))
report("Grayscale + binarize (Otsu)", predict(binary))

print("\n" + "=" * 60)
print("DEGRADED IMAGE (simulated phone-photo conditions)")
print("=" * 60 + "\n")

raw_d = load_image("data/test_samples/iam_sample_degraded.png")
gray_d = to_grayscale(raw_d)
contrast_d = enhance_contrast(gray_d)
denoised_d = denoise(gray_d)
deskewed_d = deskew(gray_d)
binary_d = binarize(gray_d)

report("Degraded - Raw (no preprocessing)", predict(raw_d))
report("Degraded - Grayscale only", predict(gray_d))
report("Degraded - Grayscale + contrast enhancement", predict(contrast_d))
report("Degraded - Grayscale + denoise", predict(denoised_d))
report("Degraded - Grayscale + deskew", predict(deskewed_d))
report("Degraded - Grayscale + binarize (Otsu)", predict(binary_d))

print("\n" + "=" * 60)
print("DEFAULT PIPELINE (grayscale + deskew)")
print("=" * 60 + "\n")

default_result = get_default_pipeline(load_image("data/test_samples/iam_sample.png"))
report("Default pipeline on clean image", predict(default_result))