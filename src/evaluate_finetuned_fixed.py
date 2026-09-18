import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from jiwer import cer
import time

model_path = "./trocr_finetuned_final"
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)

# Fix the generation config bugs found during diagnosis
model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id  # 0, not 2
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id            # 2 (already correct, set explicitly for clarity)
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id            # 1
model.generation_config.max_length = 96  # match MAX_LABEL_LENGTH used in training

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Using device: {device}")

dataset = load_dataset("Teklia/IAM-line", split="test", streaming=True)
NUM_SAMPLES = 300
predictions, references = [], []
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
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{NUM_SAMPLES} samples ({time.time()-start_time:.1f}s elapsed)")

overall_cer = cer(references, predictions)
print(f"\nEvaluated {len(references)} samples from IAM test split")
print(f"Fine-tuned model CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
print(f"Baseline (pretrained) CER was: 5.56%")