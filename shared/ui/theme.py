"""
Shared Visual Design System: Dark Luxury Cyber-Aesthetic, Frosted Glassmorphism,
and Ambient Glowing Micro-Animations.

Color Harmony:
- Canvas: Deep Obsidian (#090D16)
- Surface Glass: Frosted Navy (rgba(15, 23, 42, 0.75))
- Document A: Electric Indigo / Cyan (#38BDF8 / #6366F1)
- Document B: Cyber Magenta / Rose (#F43F5E / #EC4899)
- Overlap / Hybrid: Radiant Violet (#A855F7 / #8B5CF6)
- Clean / Safe: Radiant Emerald (#10B981)
- Warning / Review: Radiant Amber (#F59E0B)
- Critical: Neon Crimson (#EF4444)
"""

import html
import streamlit as st
import plotly.graph_objects as go

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --canvas: #090D16;
  --surface: #0F172A;
  --surface-glass: rgba(17, 24, 39, 0.72);
  --ink: #F8FAFC;
  --muted: #94A3B8;
  --line: rgba(255, 255, 255, 0.09);
  --line-focus: rgba(139, 92, 246, 0.5);
  --doc-a: #38BDF8;
  --doc-a-glow: rgba(56, 189, 248, 0.25);
  --doc-b: #F43F5E;
  --doc-b-glow: rgba(244, 63, 94, 0.25);
  --doc-overlap: #A855F7;
  --doc-overlap-glow: rgba(168, 85, 247, 0.3);
  --emerald: #10B981;
  --amber: #F59E0B;
  --crimson: #EF4444;
}

/* Background Atmosphere with Ambient Glowing Neon Orbs */
.stApp {
  background-color: var(--canvas) !important;
  background-image: 
    radial-gradient(circle at 12% 15%, rgba(56, 189, 248, 0.12) 0px, transparent 40%),
    radial-gradient(circle at 88% 20%, rgba(244, 63, 94, 0.12) 0px, transparent 40%),
    radial-gradient(circle at 50% 85%, rgba(168, 85, 247, 0.14) 0px, transparent 45%),
    linear-gradient(180deg, #090D16 0%, #0B1120 100%) !important;
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
  color: var(--ink) !important;
}

.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
  color: var(--ink);
}

.stApp h1 {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  color: #FFFFFF !important;
  text-shadow: 0 0 30px rgba(168, 85, 247, 0.3) !important;
}

.stApp h2 {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1.5rem !important;
  letter-spacing: -0.02em !important;
  color: #FFFFFF !important;
  margin-top: 1.4rem !important;
}

.stApp h3 {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1.2rem !important;
  color: #F1F5F9 !important;
}

.block-container {
  max-width: 1200px !important;
  padding-top: 2rem !important;
  padding-bottom: 4rem !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
  background: rgba(11, 17, 32, 0.85) !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  border-right: 1px solid var(--line) !important;
  box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
}

/* Buttons: Neon Multi-Stop Gradient with Soft Ambient Glow */
.stApp .stButton > button {
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%) !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 14px !important;
  padding: 0.78rem 1.8rem !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: -0.01em !important;
  box-shadow: 0 0 25px rgba(124, 58, 237, 0.45), 0 8px 16px -4px rgba(0, 0, 0, 0.5) !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stApp .stButton > button:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 0 35px rgba(236, 72, 153, 0.55), 0 12px 24px -4px rgba(0, 0, 0, 0.6) !important;
  color: #FFFFFF !important;
}

.stApp .stButton > button:active {
  transform: translateY(1px) scale(0.99) !important;
}

.stApp .stButton > button p {
  color: #FFFFFF !important;
  font-weight: 700 !important;
}

/* Download Buttons */
.stApp .stDownloadButton > button {
  background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
  color: #F8FAFC !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 14px !important;
  padding: 0.78rem 1.6rem !important;
  font-weight: 700 !important;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
  transition: all 0.25s ease !important;
}

.stApp .stDownloadButton > button:hover {
  transform: translateY(-2px) !important;
  border-color: rgba(168, 85, 247, 0.5) !important;
  box-shadow: 0 0 25px rgba(168, 85, 247, 0.35) !important;
  color: #FFFFFF !important;
}

/* File Upload Dropzones */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(15, 23, 42, 0.65) !important;
  border: 2px dashed rgba(255, 255, 255, 0.15) !important;
  border-radius: 18px !important;
  padding: 2rem !important;
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3) !important;
  transition: all 0.25s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #A855F7 !important;
  background: rgba(168, 85, 247, 0.05) !important;
  box-shadow: 0 0 25px rgba(168, 85, 247, 0.2) !important;
}

/* Dark Mode Expanders */
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  background: rgba(15, 23, 42, 0.65) !important;
  backdrop-filter: blur(16px) !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25) !important;
  margin-bottom: 14px !important;
  overflow: hidden !important;
}

[data-testid="stExpander"] summary {
  font-weight: 700 !important;
  color: #F1F5F9 !important;
  padding: 14px 20px !important;
}

/* Text Inputs and Text Areas */
.stApp textarea, .stApp input[type="text"], .stApp input[type="password"] {
  background: rgba(15, 23, 42, 0.8) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  color: #F8FAFC !important;
  border-radius: 12px !important;
  box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.3) !important;
}

.stApp textarea:focus, .stApp input:focus {
  border-color: #A855F7 !important;
  box-shadow: 0 0 15px rgba(168, 85, 247, 0.3) !important;
}

/* Animations */
@keyframes floatOrbA {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(-4px, -4px) scale(1.06); }
}

@keyframes floatOrbB {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(4px, 3px) scale(1.06); }
}

@keyframes neonPulse {
  0%, 100% { opacity: 0.85; filter: drop-shadow(0 0 12px rgba(168, 85, 247, 0.4)); }
  50% { opacity: 1; filter: drop-shadow(0 0 22px rgba(168, 85, 247, 0.8)); }
}

@keyframes cardEnter {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Hero Section */
.si-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  background: rgba(17, 24, 39, 0.72);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 22px;
  padding: 26px 34px;
  margin-bottom: 26px;
  box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
  animation: cardEnter 0.4s ease-out;
}

.si-hero-left {
  display: flex;
  align-items: center;
  gap: 26px;
}

.si-venn {
  position: relative;
  width: 86px;
  height: 54px;
  flex: none;
}

.si-venn span {
  position: absolute;
  top: 0;
  width: 54px;
  height: 54px;
  border-radius: 50%;
  mix-blend-mode: screen;
  transition: all 0.3s ease;
}

.si-a {
  left: 0;
  background: radial-gradient(circle, #38BDF8 20%, #2563EB 80%);
  box-shadow: 0 0 25px rgba(56, 189, 248, 0.6);
  animation: floatOrbA 4.5s ease-in-out infinite;
}

.si-b {
  left: 32px;
  background: radial-gradient(circle, #F43F5E 20%, #BE185D 80%);
  box-shadow: 0 0 25px rgba(244, 63, 94, 0.6);
  animation: floatOrbB 4.2s ease-in-out infinite;
}

.si-hero-title {
  font-family: 'Outfit', sans-serif;
  font-size: 2.3rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.1;
  color: #FFFFFF;
}

.si-hero-sub {
  font-size: 0.98rem;
  color: var(--muted);
  margin-top: 6px;
  line-height: 1.5;
}

/* Verdict Card */
.si-verdict {
  background: rgba(17, 24, 39, 0.75);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 22px;
  padding: 26px 30px;
  margin: 16px 0 24px 0;
  box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
  animation: cardEnter 0.35s ease-out;
}

.si-verdict-row {
  display: flex;
  align-items: center;
  gap: 28px;
}

.si-score-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 140px;
  padding: 16px 20px;
  border-radius: 18px;
  background: rgba(10, 15, 26, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.si-score {
  font-family: 'Outfit', sans-serif;
  font-size: 3.6rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
  color: #FFFFFF;
  font-variant-numeric: tabular-nums;
  text-shadow: 0 0 20px rgba(168, 85, 247, 0.5);
}

.si-score-lbl {
  font-size: 0.75rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #94A3B8;
  margin-top: 6px;
}

.si-verdict-badge {
  display: inline-block;
  font-size: 0.8rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 5px 14px;
  border-radius: 999px;
  margin-bottom: 8px;
}

.si-verdict-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.45rem;
  font-weight: 800;
  color: #FFFFFF;
  letter-spacing: -0.02em;
}

.si-verdict-note {
  font-size: 0.94rem;
  color: var(--muted);
  margin-top: 6px;
  line-height: 1.5;
}

/* Neon Meter Track */
.si-track {
  position: relative;
  margin-top: 26px;
}

.si-segs {
  display: flex;
  height: 14px;
  border-radius: 999px;
  overflow: hidden;
  box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.6);
  background: #0F172A;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.si-seg {
  flex: 1;
}

.si-marker {
  position: absolute;
  top: -5px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #FFFFFF;
  border: 4px solid #A855F7;
  transform: translateX(-50%);
  box-shadow: 0 0 15px #A855F7, 0 4px 10px rgba(0, 0, 0, 0.5);
  transition: left 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.si-labels {
  display: flex;
  margin-top: 10px;
}

.si-labels span {
  flex: 1;
  text-align: center;
  font-size: 0.8rem;
  font-weight: 700;
  color: #94A3B8;
  letter-spacing: 0.04em;
}

/* Stat Cards (Miller's Law) */
.si-stat {
  background: rgba(17, 24, 39, 0.72);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.09);
  border-radius: 18px;
  padding: 20px 22px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  transition: all 0.25s ease;
}

.si-stat:hover {
  transform: translateY(-3px);
  border-color: rgba(168, 85, 247, 0.4);
  box-shadow: 0 0 25px rgba(168, 85, 247, 0.2), 0 16px 36px rgba(0, 0, 0, 0.45);
}

.si-stat-label {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #94A3B8;
  margin-bottom: 6px;
}

.si-stat-value {
  font-family: 'Outfit', sans-serif;
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #FFFFFF;
  font-variant-numeric: tabular-nums;
  text-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
}

/* AI Forensic Card */
.si-ai-box {
  background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(17, 24, 39, 0.8) 100%);
  border: 1px solid rgba(168, 85, 247, 0.35);
  border-radius: 20px;
  padding: 24px 28px;
  margin: 18px 0;
  box-shadow: 0 0 30px rgba(168, 85, 247, 0.15), 0 15px 35px rgba(0, 0, 0, 0.4);
}

.si-ai-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.si-ai-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'Outfit', sans-serif;
  font-size: 1.15rem;
  font-weight: 800;
  color: #C084FC;
  letter-spacing: -0.01em;
  text-shadow: 0 0 15px rgba(192, 132, 252, 0.4);
}

.si-ai-source-badge {
  font-size: 0.75rem;
  font-weight: 800;
  background: rgba(168, 85, 247, 0.2);
  border: 1px solid rgba(168, 85, 247, 0.4);
  color: #E9D5FF;
  padding: 4px 12px;
  border-radius: 999px;
  box-shadow: 0 0 12px rgba(168, 85, 247, 0.2);
}

.si-ai-text {
  font-size: 0.98rem;
  color: #F1F5F9;
  line-height: 1.65;
}
</style>
"""

_LEVELS = [
    ("Minimal", "#1E1B4B", "#6366F1"),
    ("Low", "#312E81", "#818CF8"),
    ("Moderate", "#4338CA", "#A5B4FC"),
    ("High", "#7C3AED", "#C084FC"),
]


def _html(markup: str) -> str:
    return "".join(line.strip() for line in markup.splitlines())


def apply_theme():
    """Injects core dark luxury cyber design system. Call at top of page."""
    st.markdown(_html(_CSS), unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, badge_text: str = None):
    badge_html = f'<div style="display:inline-flex;align-items:center;gap:6px;background:rgba(168,85,247,0.2);border:1px solid rgba(168,85,247,0.4);color:#E9D5FF;font-size:0.75rem;font-weight:800;letter-spacing:0.05em;padding:4px 12px;border-radius:999px;margin-bottom:8px;box-shadow:0 0 15px rgba(168,85,247,0.3);">✨ {html.escape(badge_text)}</div>' if badge_text else ""
    st.markdown(
        _html(
            f"""
            <div class="si-hero">
              <div class="si-hero-left">
                <div class="si-venn"><span class="si-a"></span><span class="si-b"></span></div>
                <div>
                  {badge_html}
                  <div class="si-hero-title">{html.escape(title)}</div>
                  <div class="si-hero-sub">{html.escape(subtitle)}</div>
                </div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_category_banner(category: str, disclaimer: str = None, score: float = None):
    upper = category.upper()
    if "HIGH" in upper:
        tier_color = "#EF4444"
        tier_bg = "rgba(239, 68, 68, 0.2)"
        tier_label = "CRITICAL REVIEW REQUIRED"
        glow = "rgba(239, 68, 68, 0.35)"
    elif "MODERATE" in upper:
        tier_color = "#F59E0B"
        tier_bg = "rgba(245, 158, 11, 0.2)"
        tier_label = "REVIEW RECOMMENDED"
        glow = "rgba(245, 158, 11, 0.35)"
    elif "LOW" in upper:
        tier_color = "#38BDF8"
        tier_bg = "rgba(56, 189, 248, 0.2)"
        tier_label = "LOW SIMILARITY"
        glow = "rgba(56, 189, 248, 0.35)"
    else:
        tier_color = "#10B981"
        tier_bg = "rgba(16, 185, 129, 0.2)"
        tier_label = "INDEPENDENT"
        glow = "rgba(16, 185, 129, 0.35)"

    note = f'<div class="si-verdict-note">{html.escape(disclaimer)}</div>' if disclaimer else ""

    score_html = ""
    meter_html = ""
    if score is not None:
        shown = max(0.0, min(1.0, score))
        score_html = f"""
        <div class="si-score-box" style="border-color:{tier_color}44;box-shadow:0 0 25px {glow};">
          <div class="si-score" style="text-shadow:0 0 20px {tier_color};">{shown:.0%}</div>
          <div class="si-score-lbl">Similarity</div>
        </div>
        """
        segments = "".join(
            f'<div class="si-seg" style="background:{color};"></div>'
            for _, color, _ in _LEVELS
        )
        labels = "".join(f"<span>{name}</span>" for name, _, _ in _LEVELS)
        meter_html = f"""
        <div class="si-track">
          <div class="si-segs">{segments}</div>
          <div class="si-marker" style="left:{shown * 100:.1f}%;border-color:{tier_color};box-shadow:0 0 15px {tier_color};"></div>
        </div>
        <div class="si-labels">{labels}</div>
        """

    st.markdown(
        _html(
            f"""
            <div class="si-verdict" style="border-color:{tier_color}33;">
              <div class="si-verdict-row">
                {score_html}
                <div style="flex:1;">
                  <span class="si-verdict-badge" style="background:{tier_bg};color:{tier_color};border:1.5px solid {tier_color};box-shadow:0 0 15px {glow};">
                    {tier_label}
                  </span>
                  <div class="si-verdict-title">{html.escape(category)}</div>
                  {note}
                </div>
              </div>
              {meter_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_evidence_tickets(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                _html(
                    f"""
                    <div class="si-stat">
                      <div class="si-stat-label">{html.escape(label)}</div>
                      <div class="si-stat-value">{html.escape(value)}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )


def render_ai_box(explanation: str, source: str = "Google Gemini", status: str = "success"):
    icon = "⚡" if "Deterministic" in source else "✨"
    st.markdown(
        _html(
            f"""
            <div class="si-ai-box">
              <div class="si-ai-header">
                <div class="si-ai-title">{icon} AI Forensic Intelligence Summary</div>
                <div class="si-ai-source-badge">{html.escape(source)}</div>
              </div>
              <div class="si-ai-text">{html.escape(explanation)}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_radar_chart(hybrid_result: dict):
    categories = ['TF-IDF Cosine', 'N-gram Overlap', 'Jaccard Index', 'Semantic Cosine', 'Overall Hybrid']
    values = [
        hybrid_result.get('tfidf', 0.0),
        hybrid_result.get('ngram_overlap', 0.0),
        hybrid_result.get('jaccard', 0.0),
        hybrid_result.get('semantic', 0.0),
        hybrid_result.get('hybrid_score', 0.0),
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(168, 85, 247, 0.28)',
        line=dict(color='#C084FC', width=3),
        marker=dict(size=8, color='#38BDF8', line=dict(color='#FFFFFF', width=1.5)),
        name='Similarity Metrics',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                linecolor='rgba(255,255,255,0.15)',
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#94A3B8', size=10),
            ),
            angularaxis=dict(
                linecolor='rgba(255,255,255,0.2)',
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#F1F5F9', size=11, family='Outfit'),
            ),
            bgcolor='rgba(15, 23, 42, 0.5)',
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_sidebar_info():
    """Renders real-time compute device status in the sidebar."""
    import torch
    
    cuda_status = f"CUDA Active ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU Mode"
    cuda_color = "#10B981" if torch.cuda.is_available() else "#94A3B8"

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        _html(
            f"""
            <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.1);border-radius:14px;padding:14px 16px;box-shadow:0 4px 14px rgba(0,0,0,0.3);">
              <div style="font-size:0.75rem;font-weight:800;text-transform:uppercase;color:#94A3B8;letter-spacing:0.06em;margin-bottom:8px;">System Engine</div>
              <div style="display:flex;align-items:center;gap:7px;font-size:0.86rem;font-weight:700;color:{cuda_color};margin-bottom:6px;">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{cuda_color};box-shadow:0 0 8px {cuda_color};"></span>
                {cuda_status}
              </div>
              <div style="font-size:0.78rem;color:#64748B;margin-top:6px;">
                TrOCR Large &bull; all-MiniLM-L6-v2 &bull; Hybrid Scorer
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )