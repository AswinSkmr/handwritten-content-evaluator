from transformers import VisionEncoderDecoderModel
import torch

model = VisionEncoderDecoderModel.from_pretrained("./trocr_finetuned_final")

print("tie_word_embeddings config:", model.decoder.config.tie_word_embeddings)

emb_weight = model.decoder.model.decoder.embed_tokens.weight
out_weight = model.decoder.output_projection.weight

print("Embedding weight shape:", emb_weight.shape)
print("Output projection weight shape:", out_weight.shape)
print("Are they identical tensors (properly tied)?", torch.equal(emb_weight, out_weight))
print("Output projection weight sample values:", out_weight[0, :5])
print("Output projection weight std (random init would be small, e.g. ~0.02):", out_weight.std().item())