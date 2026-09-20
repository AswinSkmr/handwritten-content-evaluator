"""
Pulls a single sample image from the IAM test split and saves it to
disk, so we have a concrete image to test M9 preprocessing steps
against without needing to stream the dataset every time.
"""

from datasets import load_dataset

dataset = load_dataset("Teklia/IAM-line", split="test", streaming=True)
sample = next(iter(dataset.take(1)))

image = sample["image"].convert("RGB")
true_text = sample["text"]

image.save("data/test_samples/iam_sample.png")

print(f"Saved image to data/test_samples/iam_sample.png")
print(f"Ground truth text: {true_text}")