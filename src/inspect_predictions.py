import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset

model_path = "./trocr_finetuned_final"
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)

model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.max_length = 96

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

dataset = load_dataset("Teklia/IAM-line", split="test", streaming=True)

for i, sample in enumerate(dataset.take(10)):
    image = sample["image"].convert("RGB")
    true_text = sample["text"]
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, num_beams=4, no_repeat_ngram_size=3)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"\n--- Sample {i+1} ---")
    print(f"TRUE:      {true_text}")
    print(f"PREDICTED: {predicted_text}")