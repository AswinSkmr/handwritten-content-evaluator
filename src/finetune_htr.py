"""
Milestone 1 - Stage E: Fine-tune the pretrained TrOCR model on IAM's
training split, monitoring CER on the validation split after each
epoch. We never touch the test split here -- that's reserved for
the final, one-time comparison against our 5.56% baseline.
"""

import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
from jiwer import cer as compute_cer
import numpy as np

# --- Load model and processor ---
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Required config for TrOCR generation during evaluation -- tells the
# model which token marks the start/end of a generated sequence.
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Also fix the separate generation_config (this is what .generate() actually
# uses in newer Transformers versions) -- without this, eval during training
# was using the wrong start token and a too-short max_length, corrupting
# every eval_cer we saw last run even though the model was training fine.
model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
# --- Load data ---
train_dataset = load_dataset("Teklia/IAM-line", split="train")
val_dataset = load_dataset("Teklia/IAM-line", split="validation")

# --- Preprocessing function ---
# Converts a raw (image, text) pair into what the model expects:
# pixel_values (the processed image tensor) and labels (tokenized text,
# padded to a fixed max length).
MAX_LABEL_LENGTH = 96  # a bit above our observed max of 80 characters

def preprocess(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        batch["text"],
        padding="max_length",
        max_length=MAX_LABEL_LENGTH,
        truncation=True,
    ).input_ids

    # Replace padding token id with -100 so the loss function ignores
    # padded positions when computing loss (standard practice -- padding
    # shouldn't count as something the model needs to "predict").
    labels = [
        [(l if l != processor.tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]

    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    return batch

print("Preprocessing training data...")
train_dataset = train_dataset.map(preprocess, batched=True, batch_size=8,
                                   remove_columns=train_dataset.column_names)
print("Preprocessing validation data...")
val_dataset = val_dataset.map(preprocess, batched=True, batch_size=8,
                               remove_columns=val_dataset.column_names)

train_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
val_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

# --- Metric function: CER on validation predictions ---
def compute_metrics(eval_pred):
    pred_ids = eval_pred.predictions
    label_ids = eval_pred.label_ids

    # Undo the -100 replacement on BOTH predictions and labels.
    # Predictions get padded with -100 too, when the Trainer aligns
    # variable-length generated sequences across evaluation batches.
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

    return {"cer": compute_cer(label_texts, pred_texts)}

# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetuned_checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=96,      # was defaulting to 20 -- fixes truncation
    generation_num_beams=4,        # beam search reduces repetition loops
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Training complete. Saving final model...")
model.save_pretrained("./trocr_finetuned_final")
processor.save_pretrained("./trocr_finetuned_final")
print("Saved to ./trocr_finetuned_final")