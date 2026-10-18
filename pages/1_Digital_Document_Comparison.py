import tempfile
import os

import streamlit as st

from pipeline_a_digital.dispatcher import extract_text
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
    page_title="Digital Document Comparison | Similarity Intelligence",
    page_icon="⚖️",
    layout="wide",
)
apply_theme()

render_hero(
    "Digital Document Comparison",
    "Multi-dimensional lexical + semantic similarity analysis for PDF, DOCX, and TXT submissions",
)

with st.sidebar:
    render_sidebar_info()

effective_api_key = os.environ.get("GEMINI_API_KEY")

def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

# ── Quick Demo Presets ───────────────────────────────────────────────────────
DEMO_PRESETS = {
    "⚡ Demo 1: High Similarity (Verbatim Assignment Copy)": {
        "doc_a_name": "Student1_Assignment_CloudComputing.txt",
        "doc_b_name": "Student2_Assignment_CloudComputing.txt",
        "text_a": (
            "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software. "
            "Instead of buying and maintaining physical data centers, organizations rent access to storage and applications from a cloud provider. "
            "A key benefit of cloud computing is rapid elasticity, allowing workloads to scale automatically with demand spikes. "
            "Virtualization technology forms the backbone of modern multi-tenant cloud architectures."
        ),
        "text_b": (
            "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software. "
            "Rather than buying and maintaining physical data centers, organizations rent access to storage and applications from a cloud provider. "
            "A core benefit of cloud computing is rapid elasticity, which allows workloads to scale automatically with demand spikes. "
            "Virtualization technology forms the backbone of modern multi-tenant cloud architectures."
        ),
    },
    "⚡ Demo 2: Moderate Similarity (Concept Paraphrase)": {
        "doc_a_name": "Student3_Thesis_NeuralNets.txt",
        "doc_b_name": "Student4_Thesis_NeuralNets.txt",
        "text_a": (
            "Artificial neural networks are computational models inspired by the biological nervous systems of animal brains. "
            "Backpropagation calculates the gradient of the loss function with respect to each model weight using the chain rule of calculus. "
            "Overfitting occurs when a machine learning model memorizes training noise instead of generalizing to unseen validation data. "
            "Dropout regularization randomly deactivates hidden units during each training forward pass to prevent co-adaptation."
        ),
        "text_b": (
            "Deep learning systems mimic the biological neural structures found in animal brains to approximate complex mathematical functions. "
            "During backpropagation, weight gradients are computed using the calculus chain rule to systematically minimize network loss. "
            "A model overfits when it captures random noise in its training split rather than learning underlying distribution patterns. "
            "To mitigate co-adaptation among feature detectors, dropout randomly zeroes out neurons during forward propagation."
        ),
    },
    "⚡ Demo 3: Minimal Similarity (Independent Submissions)": {
        "doc_a_name": "Student5_Essay_DistributedSystems.txt",
        "doc_b_name": "Student6_Essay_DatabaseDesign.txt",
        "text_a": (
            "The Paxos consensus algorithm guarantees safety in asynchronous distributed systems with crash-stop failures. "
            "Leader election prevents split-brain conditions across geographically dispersed datacenter clusters. "
            "By maintaining an immutable replicated log, state machine replication ensures fault-tolerant state transitions."
        ),
        "text_b": (
            "Relational database normalization organizes tables to minimize data redundancy and enforce referential integrity. "
            "Boyce-Codd Normal Form eliminates all transitive functional dependencies across primary and candidate keys. "
            "ACID transactions use multi-version concurrency control to maintain isolation without blocking read queries."
        ),
    },
}

st.markdown("### ⚡ Quick-Load Evaluation Presets")
preset_cols = st.columns([1, 1, 1])
selected_preset = None

with preset_cols[0]:
    if st.button("🔥 High Similarity Demo", use_container_width=True):
        st.session_state["active_preset"] = "⚡ Demo 1: High Similarity (Verbatim Assignment Copy)"
with preset_cols[1]:
    if st.button("🔄 Paraphrase Demo", use_container_width=True):
        st.session_state["active_preset"] = "⚡ Demo 2: Moderate Similarity (Concept Paraphrase)"
with preset_cols[2]:
    if st.button("🟢 Independent Demo", use_container_width=True):
        st.session_state["active_preset"] = "⚡ Demo 3: Minimal Similarity (Independent Submissions)"

active_preset = st.session_state.get("active_preset")
if active_preset:
    st.info(f"Loaded: **{active_preset}** — click **'Analyse & Compare Documents'** below to run the pipeline.")

st.markdown("---")

# ── Document Inputs ──────────────────────────────────────────────────────────
st.markdown("### 📄 Document Input (Upload or Review)")
col1, col2 = st.columns(2, gap="large")

text_a_ready = None
text_b_ready = None
name_a = "Document A"
name_b = "Document B"

with col1:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <span style="background:#2563EB;color:#fff;font-size:0.75rem;font-weight:800;padding:2px 8px;border-radius:4px;">DOC A</span>
          <b>Primary Submission</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    file_a = st.file_uploader(
        "Upload Document A", type=["txt", "pdf", "docx"], key="doc_a",
        label_visibility="collapsed",
    )
    if file_a:
        name_a = file_a.name
        st.success(f"📎 {file_a.name} ({file_a.size / 1024:.1f} KB)")
    elif active_preset:
        preset_data = DEMO_PRESETS[active_preset]
        name_a = preset_data["doc_a_name"]
        text_a_ready = preset_data["text_a"]
        st.caption(f"Loaded preset: `{name_a}`")

with col2:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <span style="background:#DB2777;color:#fff;font-size:0.75rem;font-weight:800;padding:2px 8px;border-radius:4px;">DOC B</span>
          <b>Reference / Comparison Submission</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    file_b = st.file_uploader(
        "Upload Document B", type=["txt", "pdf", "docx"], key="doc_b",
        label_visibility="collapsed",
    )
    if file_b:
        name_b = file_b.name
        st.success(f"📎 {file_b.name} ({file_b.size / 1024:.1f} KB)")
    elif active_preset:
        preset_data = DEMO_PRESETS[active_preset]
        name_b = preset_data["doc_b_name"]
        text_b_ready = preset_data["text_b"]
        st.caption(f"Loaded preset: `{name_b}`")

st.markdown("---")

can_compare = (file_a and file_b) or (text_a_ready and text_b_ready)

if can_compare:
    if st.button("🔍 Analyse & Compare Documents", use_container_width=True):
        path_a = path_b = None
        try:
            # 1. Text Extraction
            with st.spinner("📄 Extracting text contents…"):
                if file_a:
                    path_a = save_upload_to_temp(file_a)
                    text_a = extract_text(path_a)
                else:
                    text_a = text_a_ready

                if file_b:
                    path_b = save_upload_to_temp(file_b)
                    text_b = extract_text(path_b)
                else:
                    text_b = text_b_ready

            # 2. Text Preprocessing
            with st.spinner("⚙️ Normalizing and segmenting sentences…"):
                processed_a = preprocess(text_a)
                processed_b = preprocess(text_b)

            # 3. Hybrid Plagiarism Score Computation
            with st.spinner("📊 Evaluating multi-layer lexical & semantic similarity…"):
                hybrid_result = compute_hybrid_score(
                    processed_a["normalized_text"], processed_b["normalized_text"]
                )

            # 4. Forensic Narrative Explanation (Gemini LLM / Deterministic NLP fallback)
            with st.spinner("✨ Generating forensic AI review summary…"):
                llm_res = explain_similarity_detailed(
                    text_a=processed_a["normalized_text"],
                    text_b=processed_b["normalized_text"],
                    lexical_score=hybrid_result["lexical_combined"],
                    semantic_score=hybrid_result["semantic"],
                    matching_sentences=hybrid_result["matching_sentences"],
                    api_key=effective_api_key,
                )

            # Cache results in session state so user can interact without rerunning
            st.session_state["last_eval"] = {
                "name_a": name_a,
                "name_b": name_b,
                "text_a": text_a,
                "text_b": text_b,
                "processed_a": processed_a,
                "processed_b": processed_b,
                "hybrid_result": hybrid_result,
                "llm_res": llm_res,
            }

        except Exception as e:
            st.error(f"❌ Error during document comparison: {e}")
        finally:
            for p in [path_a, path_b]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

# ── Render Evaluation Results ──────────────────────────────────────────────
if "last_eval" in st.session_state:
    res = st.session_state["last_eval"]
    name_a = res["name_a"]
    name_b = res["name_b"]
    processed_a = res["processed_a"]
    processed_b = res["processed_b"]
    hybrid_result = res["hybrid_result"]
    llm_res = res["llm_res"]
    text_a = res["text_a"]
    text_b = res["text_b"]

    st.markdown("## 📊 Evaluation Results & Evidence")

    # 1. Primary Verdict Banner
    render_category_banner(
        category=hybrid_result["category"],
        disclaimer=(
            "Similarity assessment for human academic review. "
            "A deterministic hybrid score integrates lexical overlap and semantic sentence embeddings."
        ),
        score=hybrid_result["hybrid_score"],
    )

    # 2. Key Metrics Grid (Miller's Law chunking)
    matches = hybrid_result["matching_sentences"]
    render_evidence_tickets([
        ("Overall Hybrid Score", f"{hybrid_result['hybrid_score']:.1%}"),
        ("Lexical Overlap (40%)", f"{hybrid_result['lexical_combined']:.1%}"),
        ("Semantic Meaning (60%)", f"{hybrid_result['semantic']:.1%}"),
        ("Aligned Sentences", str(len(matches))),
    ])

    # 3. AI Forensic Explanation Card
    render_ai_box(
        explanation=llm_res["text"],
        source=llm_res["source"],
        status=llm_res["status"],
    )

    # 4. Multi-dimensional Analysis Tabs
    tab_radar, tab_lexical, tab_meta = st.tabs([
        "🕸️ Similarity Radar Profile",
        "📐 Lexical Breakdown (TF-IDF, N-gram, Jaccard)",
        "📊 Document Metadata & Word Count",
    ])

    with tab_radar:
        st.caption("Multi-axial radar map illustrating similarity profile across all algorithmic layers.")
        render_radar_chart(hybrid_result)

    with tab_lexical:
        render_evidence_tickets([
            ("TF-IDF Cosine", f"{hybrid_result['tfidf']:.1%}"),
            ("N-gram (3-gram) Overlap", f"{hybrid_result['ngram_overlap']:.1%}"),
            ("Jaccard Token Index", f"{hybrid_result['jaccard']:.1%}"),
        ])

    with tab_meta:
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"**Document A (`{name_a}`)**")
            st.markdown(f"- Total Characters: `{len(text_a):,}`")
            st.markdown(f"- Words: `{len(text_a.split()):,}`")
            st.markdown(f"- Sentences: `{len(segment_sentences(text_a))}`")
        with m_col2:
            st.markdown(f"**Document B (`{name_b}`)**")
            st.markdown(f"- Total Characters: `{len(text_b):,}`")
            st.markdown(f"- Words: `{len(text_b.split()):,}`")
            st.markdown(f"- Sentences: `{len(segment_sentences(text_b))}`")

    st.markdown("---")

    # 5. Plagiarism Highlight View
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
                    doc_label=f"DOCUMENT A — {name_a}",
                    accent_color="#2563EB",
                ),
                unsafe_allow_html=True,
            )
        with hl_col2:
            st.markdown(
                build_highlight_html(
                    full_text=processed_b["normalized_text"],
                    matches=matches,
                    doc_key="b",
                    doc_label=f"DOCUMENT B — {name_b}",
                    accent_color="#DB2777",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("### 📋 Aligned Match Index")
        st.caption("Sentence pairs ranked by similarity score for granular academic scrutiny:")
        st.markdown(build_match_index_html(matches), unsafe_allow_html=True)
    else:
        st.info("No individual sentence matches detected above the 50% threshold.")

    st.markdown("---")

    # 6. PDF Evidence Report Download
    st.markdown("### 📥 Formal Evidence Report")
    st.caption("Generate an audit-ready PDF summary containing all scores, aligned sentences, and AI forensic reasoning.")
    
    report_filename = f"similarity_report_{name_a[:15]}_vs_{name_b[:15]}.pdf".replace(" ", "_")
    report_path = os.path.join("data", report_filename)
    
    generate_pdf_report(
        hybrid_result,
        report_path,
        doc_a_name=name_a,
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

    # 7. Raw Extracted Text Expander
    with st.expander("🔎 View Raw Extracted Text"):
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(f"**Document A (`{name_a}`)**")
            st.text_area("", text_a, height=240, key="raw_a", label_visibility="collapsed")
        with r_col2:
            st.markdown(f"**Document B (`{name_b}`)**")
            st.text_area("", text_b, height=240, key="raw_b", label_visibility="collapsed")

else:
    st.info("👆 Upload two documents or click one of the **Quick-Load Evaluation Presets** above to start.")