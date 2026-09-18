import tempfile
import os

import streamlit as st

from pipeline_a_digital.dispatcher import extract_text
from shared.preprocessing import preprocess
from shared.similarity.tfidf import compute_tfidf_similarity
from shared.similarity.ngram import compute_ngram_overlap
from shared.similarity.jaccard import compute_jaccard_similarity
from shared.similarity.semantic import compute_semantic_similarity
from shared.similarity.sentence_matching import find_matching_sentences

st.set_page_config(page_title="Digital Document Comparison", layout="wide")
st.title("Digital Document Comparison")
st.caption("Upload two documents (TXT, PDF, or DOCX) to compare them.")


def save_upload_to_temp(uploaded_file) -> str:
    """
    Streamlit's file_uploader gives us an in-memory file object, but
    our M4 extractors expect a file path on disk. This saves the
    upload to a temporary file and returns its path.
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Document A", type=["txt", "pdf", "docx"], key="doc_a")
with col2:
    file_b = st.file_uploader("Document B", type=["txt", "pdf", "docx"], key="doc_b")

if file_a and file_b:
    if st.button("Compare Documents"):
        try:
            path_a = save_upload_to_temp(file_a)
            path_b = save_upload_to_temp(file_b)

            with st.spinner("Extracting text..."):
                text_a = extract_text(path_a)
                text_b = extract_text(path_b)

            with st.spinner("Preprocessing..."):
                processed_a = preprocess(text_a)
                processed_b = preprocess(text_b)

            with st.spinner("Computing similarity scores..."):
                tfidf_score = compute_tfidf_similarity(
                    processed_a["normalized_text"], processed_b["normalized_text"]
                )
                ngram_score = compute_ngram_overlap(
                    processed_a["normalized_text"], processed_b["normalized_text"]
                )
                jaccard_score = compute_jaccard_similarity(
                    processed_a["normalized_text"], processed_b["normalized_text"]
                )
                semantic_score = compute_semantic_similarity(
                    processed_a["normalized_text"], processed_b["normalized_text"]
                )
                matches = find_matching_sentences(
                    processed_a["normalized_text"], processed_b["normalized_text"],
                    threshold=0.5,
                )

            st.success("Comparison complete.")

            st.markdown("### Similarity Scores")
            score_cols = st.columns(4)
            score_cols[0].metric("TF-IDF (lexical)", f"{tfidf_score:.1%}")
            score_cols[1].metric("N-gram overlap", f"{ngram_score:.1%}")
            score_cols[2].metric("Jaccard", f"{jaccard_score:.1%}")
            score_cols[3].metric("Semantic (meaning)", f"{semantic_score:.1%}")

            st.markdown("### Matching Sentences")
            if matches:
                for m in matches:
                    with st.expander(
                        f"Match (similarity: {m['similarity']:.1%})"
                    ):
                        st.write(f"**Document A:** {m['sentence_a']}")
                        st.write(f"**Document B:** {m['sentence_b']}")
            else:
                st.write("No closely matching sentences found (threshold: 50%).")

            with st.expander("View extracted text"):
                text_col1, text_col2 = st.columns(2)
                with text_col1:
                    st.text_area("Document A (extracted)", text_a, height=300)
                with text_col2:
                    st.text_area("Document B (extracted)", text_b, height=300)

        except (ValueError, RuntimeError) as e:
            st.error(f"Could not process one of the documents: {e}")
        finally:
            # Clean up temp files regardless of success/failure
            for p in [path_a, path_b]:
                if os.path.exists(p):
                    os.remove(p)
else:
    st.info("Upload both documents to enable comparison.")