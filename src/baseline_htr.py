"""
Milestone 1 - Stage B (scaled): Run the pretrained TrOCR model across
a meaningful slice of the IAM TEST split, and compute an aggregate
CER we can actually trust as a baseline.

Evaluating on 'test' (not 'train') gives us genuine held-out
performance -- this is the number future fine-tuning will be
compared against.
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from jiwer import cer
import time

# --- Load model ---
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # set model to evaluation mode (disables dropout, etc.)
print(f"Using device: {device}")

# --- Load the IAM test split ---
# streaming=True again: we don't need to download the whole dataset to
# disk, we're iterating through it once for evaluation.
dataset = load_dataset("Teklia/IAM-line", split="test", streaming=True)

# How many samples to evaluate. 300 is enough to give a statistically
# meaningful CER without taking excessively long on a first run.
# We can increase this later once we know timing.
NUM_SAMPLES = 300

predictions = []
references = []

start_time = time.time()

for i, sample in enumerate(dataset.take(NUM_SAMPLES)):
    image = sample["image"].convert("RGB")
    true_text = sample["text"]

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    predictions.append(predicted_text)
    references.append(true_text)

    # Print progress every 50 samples so we can see it's actually working,
    # not hung.
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {i + 1}/{NUM_SAMPLES} samples ({elapsed:.1f}s elapsed)")

# --- Compute overall CER ---
overall_cer = cer(references, predictions)
total_time = time.time() - start_time

print()
print(f"Evaluated {len(references)} samples from IAM test split")
print(f"Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
print(f"Total time: {total_time:.1f}s ({total_time/len(references):.2f}s per sample)")

# Save results to a file so we have a record of this baseline for later comparison
with open("data/baseline_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: Teklia/IAM-line, test split\n")
    f.write(f"Samples evaluated: {len(references)}\n")
    f.write(f"Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)\n")