import os
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline_a_digital.dispatcher import extract_text
from pipeline_b_handwritten.extract_handwriting import extract_text_from_handwriting
from shared.preprocessing import preprocess
from shared.similarity.hybrid import compute_hybrid_score
from shared.similarity.tfidf import compute_tfidf_similarity
from shared.similarity.semantic import compute_semantic_similarity
from shared.similarity.ngram import compute_ngram_overlap
from shared.similarity.jaccard import compute_jaccard_similarity
from shared.similarity.sentence_matching import find_matching_sentences
from shared.ui.theme import (
    apply_theme,
    render_hero,
    render_category_banner,
    render_evidence_tickets,
    render_sidebar_info,
)

st.set_page_config(
    page_title="60-Student Cohort Comparison | Similarity Intelligence",
    page_icon="👥",
    layout="wide",
)
apply_theme()

render_hero(
    "60-Student Cohort Comparison",
    "Pairwise cross-comparison, collusion detection, and similarity matrix analysis across 60 student submissions",
)

with st.sidebar:
    render_sidebar_info()

# ── 60-Student Synthetic Cohort Generator ───────────────────────────────────
def generate_60_student_cohort() -> dict[str, str]:
    """Generates 60 student submissions with varying degrees of similarity."""
    base_topics = [
        "Cloud computing provides scalable virtualized resources including servers, storage, and databases over the internet. Rapid elasticity enables systems to dynamically provision capacity during peak traffic. Hypervisors abstract physical hardware to maintain isolation in multi-tenant environments.",
        "Artificial neural networks emulate biological nervous architectures to perform pattern recognition. Backpropagation optimizes weights by computing loss gradients through reverse application of the calculus chain rule. Dropout regularization randomly zeroes neuron activations during training.",
        "The Raft consensus algorithm facilitates distributed state machine replication across server clusters. Heartbeat timers trigger leader elections when follower nodes stop receiving periodic heartbeats. Log entries are committed only when safely synchronized to a majority of nodes.",
        "Relational database design relies on normalization to eliminate insertion, update, and deletion anomalies. Third Normal Form dictates that every non-key column must depend directly on the primary key. Foreign key constraints enforce referential integrity across relational tables.",
        "Microservices architecture decomposes applications into loosely coupled independently deployable services. Communication typically occurs over REST APIs or event-driven message brokers. Container orchestration platforms automate service discovery, load balancing, and failure recovery.",
        "Public key cryptography enables secure communication over insecure channels using asymmetric key pairs. Digital signatures verify authenticity and integrity through hash verification. Certificate authorities validate identity claims within public key infrastructure.",
    ]

    paraphrase_topics = [
        "Cloud architectures deliver elastic computing infrastructure, remote storage, and managed databases on demand. Systems scale compute capacity dynamically to accommodate fluctuating network workloads. Virtualization hypervisors isolate guest operating systems inside multi-tenant datacenters.",
        "Deep learning frameworks simulate biological neural structures to approximate complex mathematical functions. Weight optimization uses gradient descent with the chain rule to minimize error loss across iterations. Deactivating random neurons via dropout prevents feature co-adaptation.",
        "Fault-tolerant distributed state replication is achieved using the Raft consensus protocol. When heartbeats from the primary coordinator cease, followers initiate leadership elections. State transitions commit exclusively when acknowledged by a strict quorum.",
        "Database normalization structures relational tables to prevent data redundancy and modification anomalies. Under Boyce-Codd Normal Form, every determinant must be a candidate key. Relational integrity rules guarantee consistency across connected entities.",
        "Modular distributed systems partition software logic into autonomous lightweight service units. Inter-service data flows utilize asynchronous message queues and stateless HTTP protocols. Cluster management software handles health checks and automated deployments.",
        "Asymmetric encryption mechanisms safeguard digital data transmission through mathematical public-private keypairs. Cryptographic hash functions confirm that transmitted payloads have not been altered. Trust hierarchies authenticate participants using certified credentials.",
    ]

    cohort = {}
    
    # Students 1-12: Direct duplicates and near-duplicates (Collusion clusters)
    # Pair: Student_01 <-> Student_02 (Identical topic 0)
    cohort["Student_01"] = base_topics[0]
    cohort["Student_02"] = base_topics[0]
    # Pair: Student_03 <-> Student_04 (Identical topic 1)
    cohort["Student_03"] = base_topics[1]
    cohort["Student_04"] = base_topics[1]
    # Pair: Student_05 <-> Student_06 (Identical topic 2)
    cohort["Student_05"] = base_topics[2]
    cohort["Student_06"] = base_topics[2]

    # Students 7-18: Paraphrased pairs (Semantic overlap, low lexical)
    cohort["Student_07"] = base_topics[0]
    cohort["Student_08"] = paraphrase_topics[0]
    cohort["Student_09"] = base_topics[1]
    cohort["Student_10"] = paraphrase_topics[1]
    cohort["Student_11"] = base_topics[2]
    cohort["Student_12"] = paraphrase_topics[2]
    cohort["Student_13"] = base_topics[3]
    cohort["Student_14"] = paraphrase_topics[3]
    cohort["Student_15"] = base_topics[4]
    cohort["Student_16"] = paraphrase_topics[4]
    cohort["Student_17"] = base_topics[5]
    cohort["Student_18"] = paraphrase_topics[5]

    # Students 19-60: Unique independent submissions with realistic academic topic variations
    variations = [
        "Reinforcement learning agents learn optimal decision policies by interacting with dynamic environments through trial and error.",
        "Graph neural networks process relational graph data by passing message embeddings across neighboring nodes and edges.",
        "Quantum computing harnesses superposition and entanglement to solve discrete optimization problems exponentially faster.",
        "Zero-trust security models enforce continuous identity authentication and least-privilege access across corporate networks.",
        "Compiler optimization transforms intermediate code representations to eliminate dead code and maximize instruction pipelining.",
        "Memory management in modern operating systems employs demand paging and virtual addressing to isolate process spaces.",
        "Software testing methodologies incorporate unit verification, integration testing, and automated mutation coverage.",
    ]

    for i in range(19, 61):
        sid = f"Student_{i:02d}"
        topic_idx = (i - 19) % len(base_topics)
        var_idx = (i - 19) % len(variations)
        cohort[sid] = f"{base_topics[topic_idx]} Furthermore, {variations[var_idx]}"

    return cohort


st.markdown("### 📂 Load 60-Student Cohort")
col_load, col_desc = st.columns([1, 2])

with col_load:
    if st.button("🚀 Load 60-Student Cohort (1,770 Pairs)", use_container_width=True):
        st.session_state["cohort_60"] = generate_60_student_cohort()
        st.session_state["cohort_source"] = "Simulated 60-Student Cohort (Student_01 to Student_60)"
        st.rerun()

with col_desc:
    st.caption(
        "Loads 60 student submissions across 6 academic topics, simulating exact duplicates, "
        "paraphrased collusion, and independent work across all 1,770 pairwise comparisons."
    )

st.markdown("---")

# ── Custom Batch File Uploader ──────────────────────────────────────────────
st.markdown("### 📤 Upload Student Submissions (Optional)")
uploaded_files = st.file_uploader(
    "Upload up to 60 student files (TXT, PDF, DOCX, JPG, PNG)",
    type=["txt", "pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload all assignments to run cohort-wide cross-comparison.",
)

if uploaded_files:
    custom_cohort = {}
    with st.spinner("Extracting content from uploaded files…"):
        for f in uploaded_files[:60]:
            suffix = os.path.splitext(f.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name

            try:
                if suffix in [".jpg", ".jpeg", ".png"]:
                    text = extract_text_from_handwriting(tmp_path)
                else:
                    text = extract_text(tmp_path)
                custom_cohort[f.name] = text
            except Exception as e:
                st.warning(f"Skipped {f.name}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.session_state["cohort_60"] = custom_cohort
    st.session_state["cohort_source"] = f"Custom Upload ({len(custom_cohort)} files)"

# ── Cohort Processing ────────────────────────────────────────────────────────
active_cohort = st.session_state.get("cohort_60")

if active_cohort and len(active_cohort) >= 2:
    n_students = len(active_cohort)
    total_pairs = (n_students * (n_students - 1)) // 2

    st.success(f"Ready: **{n_students} student submissions** ({total_pairs:,} pairwise combinations)")

    if st.button("🔍 Run Cohort Plagiarism Analysis", use_container_width=True):
        doc_names = list(active_cohort.keys())
        n = len(doc_names)

        with st.spinner(f"Evaluating {total_pairs:,} pairwise comparisons…"):
            # Step 1: Preprocess all documents once: O(N)
            normalized_texts = [preprocess(active_cohort[name])["normalized_text"] for name in doc_names]

            matrix = np.zeros((n, n))
            pair_records = []

            # Step 2: Compute pairwise hybrid scores
            for i in range(n):
                matrix[i][i] = 1.0
                for j in range(i + 1, n):
                    text_a = normalized_texts[i]
                    text_b = normalized_texts[j]

                    tfidf = compute_tfidf_similarity(text_a, text_b)
                    ngram = compute_ngram_overlap(text_a, text_b)
                    jaccard = compute_jaccard_similarity(text_a, text_b)
                    semantic = compute_semantic_similarity(text_a, text_b)

                    lexical = (tfidf + ngram + jaccard) / 3.0
                    hybrid = (0.4 * lexical) + (0.6 * semantic)

                    matrix[i][j] = hybrid
                    matrix[j][i] = hybrid

                    if hybrid >= 0.75:
                        category = "HIGH SIMILARITY - HUMAN REVIEW REQUIRED"
                    elif hybrid >= 0.50:
                        category = "MODERATE SIMILARITY - REVIEW RECOMMENDED"
                    elif hybrid >= 0.25:
                        category = "LOW SIMILARITY - LIKELY INDEPENDENT"
                    else:
                        category = "MINIMAL SIMILARITY - INDEPENDENT"

                    pair_records.append({
                        "Student A": doc_names[i],
                        "Student B": doc_names[j],
                        "Hybrid Similarity": hybrid,
                        "Lexical": lexical,
                        "Semantic": semantic,
                        "Category": category,
                        "text_a": text_a,
                        "text_b": text_b,
                    })

            st.session_state["cohort_results"] = {
                "doc_names": doc_names,
                "matrix": matrix,
                "pair_records": pair_records,
            }

if "cohort_results" in st.session_state:
    c_data = st.session_state["cohort_results"]
    doc_names = c_data["doc_names"]
    matrix = c_data["matrix"]
    pair_records = c_data["pair_records"]

    st.markdown("---")
    st.markdown("## 📊 Cohort Similarity Matrix Heatmap")
    st.caption("Cross-comparison matrix for all student pairs. Hover to view exact percentage:")

    clean_labels = [name.replace(".txt", "").replace(".docx", "").replace(".pdf", "") for name in doc_names]

    fig = px.imshow(
        matrix,
        x=clean_labels,
        y=clean_labels,
        color_continuous_scale=[
            (0.0, "#0B1120"),
            (0.25, "#0284C7"),
            (0.50, "#7C3AED"),
            (0.75, "#C026D3"),
            (1.0, "#F43F5E"),
        ],
        zmin=0.0,
        zmax=1.0,
        labels=dict(x="Student A", y="Student B", color="Similarity"),
    )
    fig.update_layout(
        height=660,
        margin=dict(l=40, r=40, t=30, b=40),
        font=dict(family="Outfit", size=10, color="#CBD5E1"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Suspicious Pair Ranking Table ────────────────────────────────────────
    st.markdown("### 🚨 Collusion Audit & Suspicious Pair Ranking")
    
    col_filter, col_search = st.columns([1, 1])
    with col_filter:
        min_thresh = st.slider("Minimum Similarity Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    with col_search:
        search_query = st.text_input("Filter by Student ID (e.g., Student_01)", value="")

    filtered_pairs = [
        p for p in pair_records
        if p["Hybrid Similarity"] >= min_thresh
        and (not search_query or search_query.lower() in p["Student A"].lower() or search_query.lower() in p["Student B"].lower())
    ]
    sorted_pairs = sorted(filtered_pairs, key=lambda x: x["Hybrid Similarity"], reverse=True)

    st.caption(f"Showing **{len(sorted_pairs)} flagged pairs** out of {len(pair_records):,} total combinations:")

    display_rows = []
    for p in sorted_pairs:
        sim = p["Hybrid Similarity"]
        if sim >= 0.75:
            badge = "🚨 CRITICAL REVIEW"
        elif sim >= 0.50:
            badge = "⚠️ REVIEW RECOMMENDED"
        elif sim >= 0.25:
            badge = "🔵 LOW OVERLAP"
        else:
            badge = "🟢 INDEPENDENT"

        display_rows.append({
            "Risk Tier": badge,
            "Student A": p["Student A"],
            "Student B": p["Student B"],
            "Hybrid Similarity": f"{sim:.1%}",
            "Lexical Overlap": f"{p['Lexical']:.1%}",
            "Semantic Meaning": f"{p['Semantic']:.1%}",
            "Review Category": p["Category"],
        })

    df_display = pd.DataFrame(display_rows)
    st.dataframe(df_display, use_container_width=True)

    # ── CSV Export ───────────────────────────────────────────────────────────
    st.markdown("### 📥 Export Audit Records")
    export_cols = st.columns(2)
    with export_cols[0]:
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Flagged Pairs Audit (CSV)",
            data=csv_data,
            file_name="cohort_similarity_audit.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_cols[1]:
        matrix_df = pd.DataFrame(matrix, index=doc_names, columns=doc_names)
        matrix_csv = matrix_df.to_csv().encode('utf-8')
        st.download_button(
            "⬇️ Download Full 60x60 Matrix (CSV)",
            data=matrix_csv,
            file_name="cohort_60x60_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Pair Deep-Dive ───────────────────────────────────────────────────────
    if sorted_pairs:
        st.markdown("---")
        st.markdown("### 🔬 Deep-Dive Pair Inspection")
        pair_options = [f"{p['Student A']}  ↔  {p['Student B']} ({p['Hybrid Similarity']:.0%})" for p in sorted_pairs[:50]]
        selected_idx = st.selectbox("Select pair to inspect sentence alignment:", range(len(pair_options)), format_func=lambda i: pair_options[i])

        sel_p = sorted_pairs[selected_idx]
        render_category_banner(
            category=sel_p["Category"],
            disclaimer=f"Pairwise comparison between {sel_p['Student A']} and {sel_p['Student B']}.",
            score=sel_p["Hybrid Similarity"],
        )

        matches = find_matching_sentences(sel_p["text_a"], sel_p["text_b"], threshold=0.5)
        if matches:
            st.markdown(f"**{len(matches)} Aligned Sentence Matches:**")
            for idx, m in enumerate(matches[:5], 1):
                st.markdown(
                    f"""
                    <div style="background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 18px;margin-bottom:10px;box-shadow:0 8px 20px rgba(0,0,0,0.3);">
                      <div style="font-size:0.8rem;font-weight:800;color:#C084FC;letter-spacing:0.04em;margin-bottom:6px;">MATCH #{idx} &bull; {m['similarity']:.0%} ALIGNED</div>
                      <div style="border-left:3px solid #38BDF8;padding-left:12px;margin-bottom:6px;font-size:0.92rem;color:#F1F5F9;line-height:1.5;"><b>A:</b> {m['sentence_a']}</div>
                      <div style="border-left:3px solid #F43F5E;padding-left:12px;font-size:0.92rem;color:#F1F5F9;line-height:1.5;"><b>B:</b> {m['sentence_b']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No individual sentence matches detected above the 50% threshold.")

elif not active_cohort:
    st.info("👆 Click **'Load 60-Student Cohort'** above or upload submission files to begin.")
