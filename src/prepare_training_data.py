"""
Milestone 1 - Stage E, Step 1: Load the IAM train/validation splits
with full (non-streaming) access, so we can shuffle properly during
training. This script only loads and inspects -- no training yet.
"""

from datasets import load_dataset

# Non-streaming this time: we want random access + shuffling for training.
# This downloads the full train split to disk (relatively small, ~6.5k samples).
train_dataset = load_dataset("Teklia/IAM-line", split="train")
val_dataset = load_dataset("Teklia/IAM-line", split="validation")

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Look at one example to confirm the structure matches what we saw before.
example = train_dataset[0]
print(f"\nExample entry:")
print(f"  Image size: {example['image'].size}")
print(f"  Text: {example['text']}")

# Check text length distribution -- this matters for deciding how much
# padding/truncation our tokenizer will need later.
text_lengths = [len(sample["text"]) for sample in train_dataset]
print(f"\nText length (characters) -- min: {min(text_lengths)}, "
      f"max: {max(text_lengths)}, avg: {sum(text_lengths)/len(text_lengths):.1f}")