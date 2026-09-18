from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

model_path = "./trocr_finetuned_final"
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)

print("generation_config:", model.generation_config)
print("\nconfig.max_length:", getattr(model.config, "max_length", None))
print("config.eos_token_id:", model.config.eos_token_id)
print("config.decoder_start_token_id:", model.config.decoder_start_token_id)

print("\ntokenizer.cls_token_id:", processor.tokenizer.cls_token_id)
print("tokenizer.sep_token_id:", processor.tokenizer.sep_token_id)
print("tokenizer.pad_token_id:", processor.tokenizer.pad_token_id)
print("tokenizer.bos_token_id:", processor.tokenizer.bos_token_id)
print("tokenizer.eos_token_id:", processor.tokenizer.eos_token_id)