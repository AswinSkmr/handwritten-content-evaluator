import streamlit as st

st.set_page_config(page_title="Similarity Intelligence", layout="wide")

st.title("Similarity Intelligence")
st.caption("Plagiarism Detection System for Digital and Handwritten Student Assignments")

st.markdown("""
### About this project

An MCA final-year project building a plagiarism/similarity detection
system that works on both **digital documents** (PDF/DOCX/TXT) and
**handwritten submissions** (via handwriting recognition).

Use the sidebar to navigate between pages.
""")

st.markdown("### Project Progress")

progress_data = {
    "Milestone": [
        "M0 - Project Definition",
        "M1 - Dev Environment + GitHub",
        "M4 - Digital Document Input",
        "M5 - Text Preprocessing",
        "M6 - Traditional Plagiarism Detection",
        "M7 - Semantic Similarity",
        "M8 - LLM Integration",
        "M9-M11 - Handwritten Pipeline (HTR)",
    ],
    "Status": [
        "Done", "Done", "Done", "Done", "Done", "Done",
        "In Progress", "In Progress",
    ],
}

st.table(progress_data)

st.info(
    "Digital document comparison is fully functional -- see the "
    "**Digital Document Comparison** page in the sidebar for a live demo."
)