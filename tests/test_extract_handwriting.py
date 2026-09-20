from pipeline_b_handwritten.extract_handwriting import extract_text_from_handwriting

result = extract_text_from_handwriting("data/test_samples/iam_sample.png")
print(f"Extracted text: {result}")