import tempfile
import os
import shutil

import streamlit as st
from PIL import Image

from pipeline_a_digital.dispatcher import extract_text
from pipeline_b_handwritten.extract_handwriting import extract_text_from_handwriting
from shared.preprocessing import preprocess, segment_sentences
from shared.similarity.hybrid import compute_hybrid_score
from shared.reporting.evidence_report import generate_pdf_report
from shared.reporting.highlight import (
    build_highlight_html,
    build_legend_html,
    build_match_index_html,
)
from shared.llm.explain_similarity import explain_similarity_detailed
from shared.ui.theme import (
    apply_theme,
    render_hero,
    render_category_banner,
    render_evidence_tickets,
    render_radar_chart,
    render_ai_box,
    render_sidebar_info,
)

st.set_page_config(
    page_title="Handwritten Document Analysis | Similarity Intelligence",
    page_icon="🖋️",
    layout="wide",
)
apply_theme()

render_hero(
    "Handwritten Document Analysis",
    "TrOCR Deep Learning Optical Transcription + Plagiarism Evaluation for Handwritten Submissions",
)

with st.sidebar:
    render_sidebar_info()

effective_api_key = os.environ.get("GEMINI_API_KEY")

def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

# ── Quick-Load Presets ───────────────────────────────────────────────────────
HW_PRESETS = {
    "demo_hw_1": {
        "title": "⚡ Demo 1: Real Handwritten Photo (img1.jpg vs img1.jpg)",
        "file_a": "data/test_samples/img1.jpg",
        "doc_b_type": "Handwritten image",
        "file_b": "data/test_samples/img1.jpg",
    },
    "demo_hw_2": {
        "title": "⚡ Demo 2: Cross-Submission (img1.jpg vs img2.jpg)",
        "file_a": "data/test_samples/img1.jpg",
        "doc_b_type": "Handwritten image",
        "file_b": "data/test_samples/img2.jpg",
    },
    "demo_hw_3": {
        "title": "⚡ Demo 3: IAM Form Scan vs Digital Reference",
        "file_a": "data/test_samples/iam_sample.png",
        "doc_b_type": "Digital document",
        "text_b_content": (
            "A Christian boy can marry a non-Christian girl who belongs to a different caste. "
            "Under the Special Marriage Act, religious differences do not restrict consensual adult marriage."
        ),
    },
}

st.markdown("### ⚡ Quick-Load Handwritten Presets")
preset_cols = st.columns(3)
with preset_cols[0]:
    if st.button("📝 Identical Handwritten Scan", use_container_width=True):
        st.session_state["hw_active_preset"] = "demo_hw_1"
with preset_cols[1]:
    if st.button("📑 Cross-Student Handwritten", use_container_width=True):
        st.session_state["hw_active_preset"] = "demo_hw_2"
with preset_cols[2]:
    if st.button("📋 IAM Scan vs Digital Doc", use_container_width=True):
        st.session_state["hw_active_preset"] = "demo_hw_3"

active_hw_key = st.session_state.get("hw_active_preset")
if active_hw_key:
    st.info(f"Loaded: **{HW_PRESETS[active_hw_key]['title']}** — click **'Analyse & Compare Documents'** below to run TrOCR HTR and similarity evaluation.")

st.markdown("---")

# ── Upload Section ──────────────────────────────────────────────────────────
st.markdown("### 🖋️ Upload Submissions")
col1, col2 = st.columns(2, gap="large")

file_a_path = None
file_b_path = None
text_b_direct = None
name_a = "Handwritten Submission A"
name_b = "Comparison Target B"

with col1:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <span style="background:#2563EB;color:#fff;font-size:0.75rem;font-weight:800;padding:2px 8px;border-radius:4px;">DOC A</span>
          <b>Handwritten Image (JPG/PNG)</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    file_a = st.file_uploader(
        "Upload handwritten image A", type=["jpg", "jpeg", "png"], key="hw_a",
        label_visibility="collapsed",
    )
    if file_a:
        name_a = file_a.name
        st.image(file_a, caption=f"📄 {file_a.name}", use_container_width=True)
    elif active_hw_key:
        preset_file_a = HW_PRESETS[active_hw_key]["file_a"]
        if os.path.exists(preset_file_a):
            file_a_path = preset_file_a
            name_a = os.path.basename(preset_file_a)
            st.image(preset_file_a, caption=f"📄 Preset: {name_a}", use_container_width=True)

with col2:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <span style="background:#DB2777;color:#fff;font-size:0.75rem;font-weight:800;padding:2px 8px;border-radius:4px;">DOC B</span>
          <b>Comparison Document</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    doc_b_type = st.radio(
        "Document B Type", ["Handwritten image", "Digital document"],
        horizontal=True, key="doc_b_type",
    )
    
    if doc_b_type == "Handwritten image":
        file_b = st.file_uploader(
            "Upload handwritten image B", type=["jpg", "jpeg", "png"], key="hw_b",
            label_visibility="collapsed",
        )
        if file_b:
            name_b = file_b.name
            st.image(file_b, caption=f"📄 {file_b.name}", use_container_width=True)
        elif active_hw_key and HW_PRESETS[active_hw_key]["doc_b_type"] == "Handwritten image":
            preset_file_b = HW_PRESETS[active_hw_key]["file_b"]
            if os.path.exists(preset_file_b):
                file_b_path = preset_file_b
                name_b = os.path.basename(preset_file_b)
                st.image(preset_file_b, caption=f"📄 Preset: {name_b}", use_container_width=True)
    else:
        file_b = st.file_uploader(
            "Upload digital document B", type=["txt", "pdf", "docx"], key="digital_b",
            label_visibility="collapsed",
        )
        if file_b:
            name_b = file_b.name
            st.success(f"📎 {file_b.name} ready")
        elif active_hw_key and "text_b_content" in HW_PRESETS[active_hw_key]:
            text_b_direct = HW_PRESETS[active_hw_key]["text_b_content"]
            name_b = "Digital_Reference_Assignment.txt"
            st.info(f"Loaded reference digital preset (`{name_b}`)")

st.markdown("---")

can_run_hw = (file_a or file_a_path) and (file_b or file_b_path or text_b_direct)

if can_run_hw:
    if st.button("🔍 Run TrOCR Transcription & Plagiarism Analysis", use_container_width=True):
        tmp_a = tmp_b = None
        try:
            # Resolve Document A
            if file_a:
                tmp_a = save_upload_to_temp(file_a)
                path_a = tmp_a
            else:
                path_a = file_a_path

            # 1. TrOCR Recognition on Document A
            with st.spinner("🖋️ Executing TrOCR Large line segmentation & recognition on Document A…"):
                text_a = extract_text_from_handwriting(path_a)

            # 2. Extract Document B
            with st.spinner("🖋️ Processing Document B…"):
                if text_b_direct:
                    text_b = text_b_direct
                elif doc_b_type == "Handwritten image":
                    if file_b:
                        tmp_b = save_upload_to_temp(file_b)
                        path_b = tmp_b
                    else:
                        path_b = file_b_path
                    text_b = extract_text_from_handwriting(path_b)
                else:
                    if file_b:
                        tmp_b = save_upload_to_temp(file_b)
                        path_b = tmp_b
                    text_b = extract_text(path_b)

            # 3. Preprocessing
            with st.spinner("⚙️ Normalizing extracted text…"):
                processed_a = preprocess(text_a)
                processed_b = preprocess(text_b)

            # 4. Hybrid Plagiarism Score
            with st.spinner("📊 Evaluating hybrid similarity…"):
                hybrid_result = compute_hybrid_score(
                    processed_a["normalized_text"], processed_b["normalized_text"]
                )

            # 5. AI Forensic Explanation
            with st.spinner("✨ Generating AI forensic explanation…"):
                llm_res = explain_similarity_detailed(
                    text_a=processed_a["normalized_text"],
                    text_b=processed_b["normalized_text"],
                    lexical_score=hybrid_result["lexical_combined"],
                    semantic_score=hybrid_result["semantic"],
                    matching_sentences=hybrid_result["matching_sentences"],
                    api_key=effective_api_key,
                )

            st.session_state["last_hw_eval"] = {
                "name_a": name_a,
                "name_b": name_b,
                "doc_b_type": doc_b_type,
                "text_a": text_a,
                "text_b": text_b,
                "processed_a": processed_a,
                "processed_b": processed_b,
                "hybrid_result": hybrid_result,
                "llm_res": llm_res,
            }

        except Exception as e:
            st.error(f"❌ Could not process handwritten submission: {e}")
        finally:
            for p in [tmp_a, tmp_b]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

# ── Render Handwritten Evaluation Results ──────────────────────────────────
if "last_hw_eval" in st.session_state:
    res = st.session_state["last_hw_eval"]
    name_a = res["name_a"]
    name_b = res["name_b"]
    doc_b_type = res["doc_b_type"]
    text_a = res["text_a"]
    text_b = res["text_b"]
    processed_a = res["processed_a"]
    processed_b = res["processed_b"]
    hybrid_result = res["hybrid_result"]
    llm_res = res["llm_res"]

    st.markdown("## 📊 Handwritten Plagiarism Evaluation Results")

    # Verdict Card
    render_category_banner(
        category=hybrid_result["category"],
        disclaimer=(
            "Similarity assessment for human academic review. "
            "Handwriting recognition may introduce minor transcription errors — "
            "always verify the highlighted passages against the original scan before reaching conclusions."
        ),
        score=hybrid_result["hybrid_score"],
    )

    # Key Metrics
    matches = hybrid_result["matching_sentences"]
    render_evidence_tickets([
        ("Overall Hybrid Score", f"{hybrid_result['hybrid_score']:.1%}"),
        ("Lexical Overlap (40%)", f"{hybrid_result['lexical_combined']:.1%}"),
        ("Semantic Meaning (60%)", f"{hybrid_result['semantic']:.1%}"),
        ("Aligned Sentences", str(len(matches))),
    ])

    # AI Forensic Card
    render_ai_box(
        explanation=llm_res["text"],
        source=llm_res["source"],
        status=llm_res["status"],
    )

    # Multi-dimensional Analysis Tabs
    tab_radar, tab_lexical, tab_htr = st.tabs([
        "🕸️ Similarity Radar Profile",
        "📐 Lexical Sub-metrics (TF-IDF, N-gram, Jaccard)",
        "🖋️ TrOCR Transcription & Metadata",
    ])

    with tab_radar:
        st.caption("Multi-axial radar map illustrating similarity profile across all algorithmic layers.")
        render_radar_chart(hybrid_result)

    with tab_lexical:
        render_evidence_tickets([
            ("TF-IDF Cosine", f"{hybrid_result['tfidf']:.1%}"),
            ("N-gram Overlap", f"{hybrid_result['ngram_overlap']:.1%}"),
            ("Jaccard Token Index", f"{hybrid_result['jaccard']:.1%}"),
        ])

    with tab_htr:
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown(f"**Document A: TrOCR Output (`{name_a}`)**")
            st.markdown(f"- Characters Transcribed: `{len(text_a):,}`")
            st.markdown(f"- Words Recognized: `{len(text_a.split()):,}`")
            st.markdown(f"- Sentences Formed: `{len(segment_sentences(text_a))}`")
            st.markdown(f"- Model: `microsoft/trocr-large-handwritten` (ViT Encoder + Autoregressive Decoder)")
        with h_col2:
            st.markdown(f"**Document B (`{name_b}`)**")
            st.markdown(f"- Characters: `{len(text_b):,}`")
            st.markdown(f"- Words: `{len(text_b.split()):,}`")
            st.markdown(f"- Mode: `{doc_b_type}`")

    st.markdown("---")

    # Plagiarism Highlight View
    st.markdown("### 🔦 Plagiarism Highlight View")
    st.caption(
        "Cross-document comparative viewer. Suspicious text segments are highlighted by similarity strength. "
        "Numbered badges (#1, #2...) index matching sentence pairs between Document A and Document B."
    )
    st.markdown(build_legend_html(), unsafe_allow_html=True)

    if matches:
        hl_col1, hl_col2 = st.columns(2, gap="large")
        with hl_col1:
            st.markdown(
                build_highlight_html(
                    full_text=processed_a["normalized_text"],
                    matches=matches,
                    doc_key="a",
                    doc_label=f"DOCUMENT A (HTR) — {name_a}",
                    accent_color="#2563EB",
                ),
                unsafe_allow_html=True,
            )
        with hl_col2:
            b_label = f"DOCUMENT B (HTR) — {name_b}" if doc_b_type == "Handwritten image" else f"DOCUMENT B (Digital) — {name_b}"
            st.markdown(
                build_highlight_html(
                    full_text=processed_b["normalized_text"],
                    matches=matches,
                    doc_key="b",
                    doc_label=b_label,
                    accent_color="#DB2777",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("### 📋 Aligned Match Index")
        st.markdown(build_match_index_html(matches), unsafe_allow_html=True)
    else:
        st.info("No individual sentence matches detected above the 50% threshold.")

    st.markdown("---")

    # PDF Evidence Report
    st.markdown("### 📥 Formal Evidence Report")
    report_filename = f"similarity_report_handwritten_{name_a[:12]}_vs_{name_b[:12]}.pdf".replace(" ", "_")
    report_path = os.path.join("data", report_filename)

    generate_pdf_report(
        hybrid_result,
        report_path,
        doc_a_name=f"{name_a} (Handwritten HTR)",
        doc_b_name=name_b,
        explanation=llm_res["text"],
        explanation_source=llm_res["source"],
    )

    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                "⬇️ Download Formal Evidence Report (PDF)",
                f,
                file_name=report_filename,
                mime="application/pdf",
                use_container_width=True,
            )

    # Raw Text Expander
    with st.expander("🔎 Inspect Raw Extracted & HTR Text"):
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(f"**Document A — Raw TrOCR Output** (`{name_a}`)")
            st.text_area("", text_a, height=240, key="raw_hw_a", label_visibility="collapsed")
        with r_col2:
            st.markdown(f"**Document B — Extracted Text** (`{name_b}`)")
            st.text_area("", text_b, height=240, key="raw_hw_b", label_visibility="collapsed")

else:
    st.info("👆 Upload a handwritten image or select one of the **Quick-Load Handwritten Presets** to begin.")