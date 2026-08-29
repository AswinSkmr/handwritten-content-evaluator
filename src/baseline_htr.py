"""
Milestone 1 - Stage B: Run a pretrained TrOCR model (no fine-tuning)
on our saved IAM sample images, and compute CER against the known
ground-truth transcriptions.

This establishes a baseline BEFORE any fine-tuning, so we have a
real number to compare future improvements against.
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from jiwer import cer

# --- Load the pretrained model and its matching processor ---
# The "processor" handles converting images into the tensor format
# the model expects, and later decoding the model's output back into text.
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Move the model to the GPU if available, otherwise fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# --- Load our saved samples and their ground-truth transcriptions ---
samples_dir = "data/iam_samples"
transcriptions_file = f"{samples_dir}/transcriptions.txt"

ground_truths = {}
with open(transcriptions_file, "r", encoding="utf-8") as f:
    for line in f:
        filename, text = line.strip().split("\t")
        ground_truths[filename] = text

# --- Run the model on each image and compare to ground truth ---
predictions = []
references = []

for filename, true_text in ground_truths.items():
    image_path = f"{samples_dir}/{filename}"
    image = Image.open(image_path).convert("RGB")

    # Convert the image into the tensor format TrOCR expects.
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate text. torch.no_grad() disables gradient tracking since
    # we're only doing inference, not training -- this saves memory
    # and time.
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"File: {filename}")
    print(f"  Ground truth: {true_text}")
    print(f"  Predicted:    {predicted_text}")
    print()

    predictions.append(predicted_text)
    references.append(true_text)

# --- Compute overall CER across all samples ---
overall_cer = cer(references, predictions)
print(f"Overall CER on {len(references)} samples: {overall_cer:.4f} ({overall_cer*100:.2f}%)")