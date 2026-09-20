"""
M15 - Evidence and reporting.

Takes M14's hybrid score output and produces two things:
1. A structured summary dict (for display in the Streamlit UI).
2. A downloadable PDF evidence report (for handing to a reviewer).

Per the project's core principle, this never presents a bare
percentage as a verdict -- every report leads with the category
("... - HUMAN REVIEW REQUIRED" etc.) and shows the supporting
evidence (individual scores, matching passages) a reviewer would
need to make their own judgment.
"""

from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)


def summarize_evidence(hybrid_result: dict, longest_match: str = None) -> dict:
    """
    Extracts a clean, display-ready summary from a compute_hybrid_score()
    result. Computes the longest matching phrase across all matched
    sentence pairs if not already provided.
    """
    matches = hybrid_result["matching_sentences"]

    if longest_match is None and matches:
        longest_match = max(
            (m["sentence_a"] for m in matches), key=len, default=""
        )

    return {
        "overall_similarity": hybrid_result["hybrid_score"],
        "lexical_similarity": hybrid_result["lexical_combined"],
        "semantic_similarity": hybrid_result["semantic"],
        "category": hybrid_result["category"],
        "matching_sentence_count": len(matches),
        "longest_matching_phrase": longest_match or "(none found)",
        "matches": matches,
    }


def generate_pdf_report(
    hybrid_result: dict, output_path: str,
    doc_a_name: str = "Document A", doc_b_name: str = "Document B",
) -> None:
    """
    Writes a structured PDF evidence report to output_path, suitable
    for handing to a human reviewer. Never states a plagiarism verdict
    -- only the category label and supporting evidence.
    """
    summary = summarize_evidence(hybrid_result)

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    category_style = ParagraphStyle(
        "Category", parent=styles["Heading2"], textColor=colors.HexColor("#B91C1C"),
    )

    elements = []
    elements.append(Paragraph("Similarity Evidence Report", styles["Title"]))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]
    ))
    elements.append(Paragraph(f"{doc_a_name}  vs.  {doc_b_name}", styles["Normal"]))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph(summary["category"], category_style))
    elements.append(Spacer(1, 0.2 * inch))

    score_table_data = [
        ["Metric", "Score"],
        ["Overall (hybrid) similarity", f"{summary['overall_similarity']:.1%}"],
        ["Lexical similarity", f"{summary['lexical_similarity']:.1%}"],
        ["Semantic similarity", f"{summary['semantic_similarity']:.1%}"],
        ["Matching sentence pairs found", str(summary["matching_sentence_count"])],
    ]
    score_table = Table(score_table_data, colWidths=[3 * inch, 2 * inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("Longest Matching Phrase", styles["Heading3"]))
    elements.append(Paragraph(summary["longest_matching_phrase"], styles["Normal"]))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Matching Sentence Pairs", styles["Heading3"]))
    if summary["matches"]:
        for i, m in enumerate(summary["matches"], 1):
            elements.append(Paragraph(
                f"<b>{i}.</b> ({m['similarity']:.0%} similar)", styles["Normal"]
            ))
            elements.append(Paragraph(f"&nbsp;&nbsp;A: {m['sentence_a']}", styles["Normal"]))
            elements.append(Paragraph(f"&nbsp;&nbsp;B: {m['sentence_b']}", styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
    else:
        elements.append(Paragraph("No individual sentence matches above threshold.", styles["Normal"]))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(
        "This report presents similarity evidence for human review. "
        "It does not constitute a determination of academic misconduct.",
        styles["Italic"],
    ))

    doc.build(elements)