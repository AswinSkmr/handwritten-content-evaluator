import streamlit as st

st.set_page_config(page_title="Model Information", layout="wide")
st.title("Model & Method Information")
st.caption(
    "A transparent breakdown of what this system uses, per the "
    "project's distinction between pretrained components used in "
    "production and educational demonstrations of training concepts."
)

st.markdown("## Pretrained Models Used in Production")

st.markdown("""
| Component | Model | Role |
|---|---|---|
| Handwriting recognition (HTR) | `microsoft/trocr-base-handwritten` (pretrained, not fine-tuned) | Converts handwritten images to text |
| Semantic similarity | `sentence-transformers/all-MiniLM-L6-v2` | Generates meaning-based embeddings for paraphrase detection |
| LLM-assisted explanation | Google Gemini (`gemini-3.6-flash`), free tier | Explains *why* a flagged pair looks similar, in plain language |
""")

st.info(
    "**None of these three models were trained from scratch or "
    "fine-tuned for production use.** The TrOCR HTR model was "
    "**attempted** to be fine-tuned on the IAM dataset (twice), but "
    "both attempts degraded performance well below the pretrained "
    "baseline -- documented honestly below as a negative result, not "
    "hidden or omitted."
)

st.markdown("## Deterministic Methods (No Model Weights)")
st.markdown("""
- **TF-IDF + cosine similarity** -- word-frequency-based lexical similarity
- **N-gram overlap** -- exact phrase-sequence matching (overlap coefficient)
- **Jaccard similarity** -- set-based overlap, penalizes length mismatch
- **Sentence-level matching** -- pairwise sentence comparison for evidence
""")

st.markdown("## What the LLM Does NOT Do")
st.warning(
    "The LLM is never allowed to issue a plagiarism verdict or a "
    "percentage. Its prompt explicitly instructs it to explain only, "
    "and every score/category shown in this app comes from the "
    "deterministic hybrid scorer (M14), not the LLM. This is enforced "
    "in the prompt itself, not just by convention."
)

st.markdown("## Fine-Tuning Attempt: An Honest Negative Result")
st.markdown("""
Two attempts were made to fine-tune TrOCR on the IAM training split,
hoping to improve on the pretrained baseline's 5.56% CER:

1. **Attempt 1** (2 epochs): failed due to a generation-config bug
   (`decoder_start_token_id` mismatch, `max_length` too short),
   producing 47-53% CER.
2. **Attempt 2** (5 epochs, config bug fixed): still failed --
   best result was 46.4% CER at just 0.37 epochs in, then got
   progressively *worse* through the remaining training. Diagnosed
   as likely catastrophic forgetting from too-aggressive early
   learning rate with no warmup.

**Decision:** given a fixed project deadline and two failed
multi-hour training runs, further attempts were not pursued. The
**pretrained baseline (5.56% CER) is used in production**, and this
is documented as a limitation / future-work item rather than a
hidden failure.
""")

st.markdown("## Hybrid Scoring Weights")
st.markdown("""
The hybrid score combines lexical similarity (40% weight) and
semantic similarity (60% weight). This weighting is a **reasoned
starting point** grounded in evidence gathered during development
(semantic similarity stayed robust to HTR noise in real testing,
while lexical methods did not) -- **not weights formally tuned
against a labeled validation dataset**, since that dataset (M12) was
scaled down given project constraints. This is stated here
explicitly rather than presented as more rigorously validated than
it is.
""")
