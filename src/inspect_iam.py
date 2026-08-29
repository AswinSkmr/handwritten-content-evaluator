"""
Milestone 1 - Step 2: Load and inspect a few samples from the IAM
handwriting dataset (via the Teklia/IAM-line mirror on Hugging Face).

This script does NOT train anything. Its only job is to let us look
at real data before we write any pipeline code around it.
"""

from datasets import load_dataset
 
# Load only the 'train' split for now. The 'streaming=True' flag means
# we don't download the whole ~266MB dataset up front -- we pull
# samples on demand. This is useful right now because we only want
# to LOOK at a handful of examples, not commit to a full download yet.
dataset = load_dataset("Teklia/IAM-line", split="train", streaming=True)

# Take the first 5 samples and save their images + transcriptions
# locally so we can actually open and look at them.
output_dir = "data/iam_samples"
import os
os.makedirs(output_dir, exist_ok=True)

transcriptions = []

for i, sample in enumerate(dataset.take(5)):
    image = sample["image"]
    text = sample["text"]

    image_path = os.path.join(output_dir, f"sample_{i}.jpg")
    image.save(image_path)

    print(f"Sample {i}:")
    print(f"  Image size: {image.size}")   # (width, height) in pixels
    print(f"  Text: {text}")
    print(f"  Saved to: {image_path}")
    print()

    transcriptions.append(f"sample_{i}.jpg\t{text}")

# Save transcriptions to a small text file alongside the images
with open(os.path.join(output_dir, "transcriptions.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(transcriptions))

print(f"Done. Inspect the images in {output_dir}/")