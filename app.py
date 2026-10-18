import streamlit as st
from shared.ui.theme import apply_theme, render_hero, render_sidebar_info

st.set_page_config(
    page_title="Similarity Intelligence",
    page_icon="⚖️",
    layout="wide",
)
apply_theme()

render_hero(
    "Similarity Intelligence",
    "Multimodal plagiarism detection and similarity analysis for digital, handwritten, and 60-student cohort submissions",
)

with st.sidebar:
    render_sidebar_info()

# ── Core System Modules ──────────────────────────────────────────────────────
st.markdown("### 🚀 Core Modules")
st.caption("Select a module from the sidebar or review below to begin analysis:")

col_m1, col_m2, col_m3 = st.columns(3, gap="medium")

with col_m1:
    st.markdown(
        """
        <div style="background:rgba(17,24,39,0.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.09);border-left:5px solid #38BDF8;border-radius:18px;padding:22px;box-shadow:0 12px 30px rgba(0,0,0,0.35);height:100%;">
          <div style="font-family:'Outfit',sans-serif;font-weight:800;font-size:1.2rem;color:#FFFFFF;margin-bottom:8px;">📄 Digital Comparison</div>
          <div style="font-size:0.92rem;color:#94A3B8;line-height:1.6;margin-bottom:14px;">
            Compare two digital assignments (PDF, DOCX, or TXT). 
            Generates interactive multi-axial radar profiles, synchronized inline highlight views, and formal PDF evidence reports.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        """
        <div style="background:rgba(17,24,39,0.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.09);border-left:5px solid #F43F5E;border-radius:18px;padding:22px;box-shadow:0 12px 30px rgba(0,0,0,0.35);height:100%;">
          <div style="font-family:'Outfit',sans-serif;font-weight:800;font-size:1.2rem;color:#FFFFFF;margin-bottom:8px;">🖋️ Handwritten Analysis</div>
          <div style="font-size:0.92rem;color:#94A3B8;line-height:1.6;margin-bottom:14px;">
            Analyze photos or scans of handwritten submissions. 
            Runs TrOCR optical transcription and compares against digital or handwritten reference documents.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        """
        <div style="background:rgba(17,24,39,0.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.09);border-left:5px solid #A855F7;border-radius:18px;padding:22px;box-shadow:0 12px 30px rgba(0,0,0,0.35);height:100%;">
          <div style="font-family:'Outfit',sans-serif;font-weight:800;font-size:1.2rem;color:#FFFFFF;margin-bottom:8px;">👥 60-Student Cohort</div>
          <div style="font-size:0.92rem;color:#94A3B8;line-height:1.6;margin-bottom:14px;">
            Cross-evaluate up to 60 student submissions across all 1,770 pairwise combinations. 
            Generates similarity matrix heatmaps, collusion ranking, and CSV audit reports.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Architecture Overview ───────────────────────────────────────────────────
st.markdown("### 🏛️ Pipeline Architecture")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown(
        """
        <div style="background:rgba(17,24,39,0.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.09);border-radius:16px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,0.3);height:100%;">
          <div style="font-size:1.4rem;margin-bottom:6px;">📑</div>
          <div style="font-family:'Outfit',sans-serif;font-weight:800;color:#38BDF8;font-size:1.05rem;margin-bottom:4px;">Dual Ingestion</div>
          <div style="font-size:0.86rem;color:#94A3B8;line-height:1.5;">
            Digital files (PDF/DOCX/TXT) and handwritten scans converge into the same normalized evaluation pipeline.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p2:
    st.markdown(
        """
        <div style="background:rgba(17,24,39,0.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.09);border-radius:16px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,0.3);height:100%;">
          <div style="font-size:1.4rem;margin-bottom:6px;">⚖️</div>
          <div style="font-family:'Outfit',sans-serif;font-weight:800;color:#A855F7;font-size:1.05rem;margin-bottom:4px;">Hybrid Scoring</div>
          <div style="font-size:0.86rem;color:#94A3B8;line-height:1.5;">
            Combines deterministic lexical metrics (TF-IDF, 3-gram, Jaccard) with transformer semantic embeddings.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p3:
    st.markdown(
        """
        <div style="background:rgba(17,24,39,0.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.09);border-radius:16px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,0.3);height:100%;">
          <div style="font-size:1.4rem;margin-bottom:6px;">🔦</div>
          <div style="font-family:'Outfit',sans-serif;font-weight:800;color:#F43F5E;font-size:1.05rem;margin-bottom:4px;">Evidence-Based</div>
          <div style="font-size:0.86rem;color:#94A3B8;line-height:1.5;">
            Highlights matching passages inline with similarity strength tiers and downloadable audit reports.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )