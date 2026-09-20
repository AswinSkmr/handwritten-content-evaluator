from shared.similarity.hybrid import compute_hybrid_score
from shared.reporting.evidence_report import summarize_evidence, generate_pdf_report

text_a = 'assuredness " Bella Bella Marie " ( Parlophone ) a lively song that change\'s tempo midway .'
text_b = 'assuredness "Bella Bella Marie" (Parlophone), a lively song that changes tempo mid-way.'

result = compute_hybrid_score(text_a, text_b)

summary = summarize_evidence(result)
print("--- Structured Summary ---")
for key, value in summary.items():
    if key != "matches":
        print(f"{key}: {value}")

generate_pdf_report(
    result, "data/test_samples/evidence_report_test.pdf",
    doc_a_name="Handwritten Submission (HTR)", doc_b_name="Digital Submission",
)
print("\nPDF report generated: data/test_samples/evidence_report_test.pdf")