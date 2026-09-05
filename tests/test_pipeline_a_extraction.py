from pipeline_a_digital.extract_txt import extract_text_from_txt
from pipeline_a_digital.extract_pdf import extract_text_from_pdf
from pipeline_a_digital.extract_docx import extract_text_from_docx
from pipeline_a_digital.dispatcher import extract_text


print("--- TXT extraction test ---")
result = extract_text_from_txt("data/test_samples/sample1.txt")
print(f"Extracted text: {repr(result)}")

try:
    extract_text_from_txt("data/test_samples/empty.txt")
    print("ERROR: should have raised ValueError but didn't!")
except ValueError as e:
    print(f"Correctly caught empty file: {e}")

print("\n--- PDF extraction test ---")
try:
    pdf_result = extract_text_from_pdf("data/test_samples/sample1.pdf")
    print(f"Extracted {len(pdf_result)} characters from PDF")
    print(f"First 300 chars:\n{pdf_result[:300]}")
except (ValueError, RuntimeError) as e:
    print(f"PDF extraction failed: {e}")

print("\n--- DOCX extraction test ---")
try:
    docx_result = extract_text_from_docx("data/test_samples/sample1.docx")
    print(f"Extracted {len(docx_result)} characters from DOCX")
    print(f"First 300 chars:\n{docx_result[:300]}")
except (ValueError, RuntimeError) as e:
    print(f"DOCX extraction failed: {e}")

print("\n--- Dispatcher test ---")
for path in ["data/test_samples/sample1.txt", "data/test_samples/sample1.pdf", "data/test_samples/sample1.docx"]:
    text = extract_text(path)
    print(f"{path}: {len(text)} chars extracted")

try:
    extract_text("data/test_samples/sample1.jpg")
    print("ERROR: should have raised ValueError for unsupported extension!")
except (ValueError, FileNotFoundError) as e:
    print(f"Correctly handled unsupported/missing file: {e}")