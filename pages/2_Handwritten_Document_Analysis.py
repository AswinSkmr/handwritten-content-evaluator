import tempfile
import os

import streamlit as st

from pipeline_a_digital.dispatcher import extract_text
from pipeline_b_handwritten.extract_handwriting import extract_text_from_handwriting
from shared.preprocessing import preprocess
from shared.similarity.hybrid import compute_hybrid_score
from shared.reporting.evidence_report import generate_pdf_report

st.set_page_config(page_title="Handwritten Document Analysis", layout="wide")
st.title("Handwritten Document Analysis")
st.caption(
    "Compare a handwritten submission (image or scan) against a "
    "digital document, or against another handwritten submission."
)


def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


col1, col2 = st.columns(2)
with col1:
    st.markdown("**Document A (Handwritten)**")
    file_a = st.file_uploader(
        "Handwritten image", type=["jpg", "jpeg", "png"], key="hw_a"
    )
with col2:
    st.markdown("**Document B**")
    doc_b_type = st.radio(
        "Document B type", ["Handwritten image", "Digital document"],
        horizontal=True, key="doc_b_type",
    )
    if doc_b_type == "Handwritten image":
        file_b = st.file_uploader(
            "Handwritten image", type=["jpg", "jpeg", "png"], key="hw_b"
        )
    else:
        file_b = st.file_uploader(
            "Digital document", type=["txt", "pdf", "docx"], key="digital_b"
        )

if file_a and file_b:
    if st.button("Compare Documents"):
        path_a, path_b = None, None
        try:
            path_a = save_upload_to_temp(file_a)
            path_b = save_upload_to_temp(file_b)

            with st.spinner("Running handwriting recognition on Document A..."):
                text_a = extract_text_from_handwriting(path_a)

            with st.spinner("Processing Document B..."):
                if doc_b_type == "Handwritten image":
                    text_b = extract_text_from_handwriting(path_b)
                else:
                    text_b = extract_text(path_b)

            with st.spinner("Preprocessing..."):
                processed_a = preprocess(text_a)
                processed_b = preprocess(text_b)

            with st.spinner("Computing hybrid similarity score..."):
                hybrid_result = compute_hybrid_score(
                    processed_a["normalized_text"], processed_b["normalized_text"]
                )

            st.success("Comparison complete.")

            st.markdown(f"## {hybrid_result['category']}")
            st.caption(
                "This is a similarity assessment for human review, not a "
                "determination of academic misconduct. Handwriting "
                "recognition may introduce transcription errors -- see "
                "the extracted text below to verify."
            )

            score_cols = st.columns(3)
            score_cols[0].metric("Overall (hybrid) similarity", f"{hybrid_result['hybrid_score']:.1%}")
            score_cols[1].metric("Lexical similarity", f"{hybrid_result['lexical_combined']:.1%}")
            score_cols[2].metric("Semantic similarity", f"{hybrid_result['semantic']:.1%}")

            st.markdown("### Matching Sentences")
            matches = hybrid_result["matching_sentences"]
            if matches:
                for m in matches:
                    with st.expander(f"Match (similarity: {m['similarity']:.1%})"):
                        st.write(f"**Document A:** {m['sentence_a']}")
                        st.write(f"**Document B:** {m['sentence_b']}")
            else:
                st.write("No closely matching sentences found (threshold: 50%).")

            pdf_path = f"data/report_handwritten_{file_a.name}_vs_{file_b.name}.pdf"
            generate_pdf_report(
                hybrid_result, pdf_path,
                doc_a_name=file_a.name, doc_b_name=file_b.name,
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download Evidence Report (PDF)", f,
                    file_name="similarity_evidence_report.pdf",
                    mime="application/pdf",
                )

            with st.expander("View extracted text (verify HTR accuracy)"):
                text_col1, text_col2 = st.columns(2)
                with text_col1:
                    st.text_area("Document A (HTR output)", text_a, height=250)
                with text_col2:
                    label = "Document B (HTR output)" if doc_b_type == "Handwritten image" else "Document B (extracted)"
                    st.text_area(label, text_b, height=250)

        except (ValueError, RuntimeError) as e:
            st.error(f"Could not process one of the documents: {e}")
        finally:
            for p in [path_a, path_b]:
                if p and os.path.exists(p):
                    os.remove(p)
else:
    st.info("Upload Document A (handwritten) and Document B to enable comparison.")