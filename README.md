# Similarity Intelligence (Syme)

An MCA final-year project: a plagiarism/similarity detection system
for both **digital documents** (PDF/DOCX/TXT) and **handwritten**
**student submissions** (via handwriting recognition), built with a
hybrid lexical + semantic + LLM-assisted approach.

## Project Status

Core detection pipeline is functionally complete and tested. See
app.py (Home page) for a live milestone progress table, or the
summary below.

| Milestone | Status |
|---|---|
| M0 - Project Definition | Done |
| M1 - Dev Environment + GitHub | Done |
| M4 - Digital Document Input (TXT/PDF/DOCX) | Done |
| M5 - Text Preprocessing | Done |
| M6 - Traditional Plagiarism Detection (TF-IDF, n-gram, Jaccard, sentence matching) | Done |
| M7 - Semantic Similarity (sentence-transformers) | Done |
| M8 - LLM Integration (Gemini) | Done |
| M9 - Handwritten Image Preprocessing | Done (evidence-based: default pipeline is grayscale + deskew only) |
| M10 - Handwriting Recognition | Done (pretrained TrOCR baseline used in production; fine-tuning attempted twice, documented as a negative result - see Limitations) |
| M11 - Handwritten-to-Plagiarism Pipeline | Done |
| M14 - Hybrid Plagiarism Score | Done |
| M15 - Evidence and Reporting (PDF reports) | Done |
| M16 - Streamlit Application | Done (Home, Digital Document Comparison, Handwritten Document Analysis, Model Information) |
| M12/M13 - Validation dataset + 60-student scale test | Scaled down / pending |
| M17-M20 - Testing, performance, deployment | In progress |

## Architecture

```
Digital document (PDF/DOCX/TXT)
    -> extraction (pipeline_a_digital/)
Handwritten image (JPG/PNG)
    -> preprocessing + HTR (pipeline_b_handwritten/)
                    |
                    v
    shared/preprocessing.py   <- both pipelines converge here
                    |
                    v
    shared/similarity/   (TF-IDF, n-gram, Jaccard, semantic, hybrid)
                    |
                    v
    shared/llm/   (Gemini-assisted explanation, flagged pairs only)
                    |
                    v
    shared/reporting/   (structured evidence + PDF report)
```

Both pipelines produce plain text and feed into the **same**
similarity/reporting logic - no duplicated plagiarism-detection code
between the digital and handwritten paths.

## Tech Stack

- **App:** Streamlit
- **ML/NLP:** PyTorch, Hugging Face Transformers (transformers==4.57.6,
  pinned - see Known Issues), sentence-transformers, scikit-learn, NLTK
- **Image processing:** OpenCV, Pillow
- **LLM:** Google Gemini (gemini-3.6-flash, free tier) via google-genai
- **Document parsing:** pdfplumber, python-docx
- **Reporting:** reportlab (PDF generation)
- **Testing:** pytest-compatible test scripts under tests/

## Setup

```
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Additional one-time setup:**

1. Download NLTK's sentence tokenizer data:

```
python -c "import nltk; nltk.download('punkt_tab')"
```

2. Create a .env file in the project root with a Gemini API key
   (free, no billing required - get one at
   https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_key_here
```

## Running the App

```
streamlit run app.py
```

## Running Tests

Test scripts live under tests/ and must be run as modules from the
project root (not directly), so Python can resolve the project's
internal package imports:

```
python -m tests.test_hybrid
python -m tests.test_pipeline_convergence
```
(and so on for other test files in that folder)

## Project Structure

```
pipeline_a_digital/     Digital document extraction (TXT/PDF/DOCX)
pipeline_b_handwritten/ Handwritten image preprocessing + HTR
shared/
  preprocessing.py       Shared text normalization (both pipelines)
  similarity/            TF-IDF, n-gram, Jaccard, semantic, hybrid
  llm/                   Gemini-assisted explanation
  reporting/             Evidence summaries + PDF reports
pages/                  Streamlit multi-page UI
src/                    HTR training/evaluation/diagnostic scripts
tests/                  Test scripts (run as modules, see above)
data/                   Local data (mostly gitignored; small evidence
                        samples under data/test_samples/ are tracked)
```

## Known Issues / Limitations

- **transformers is pinned to 4.57.6.** A newer version broke
  TrOCR's tokenizer loading during development; this pin should be
  revisited if dependencies are upgraded later.
- **HTR fine-tuning was attempted twice and did not improve on the**
  **pretrained baseline** (5.56% CER on held-out IAM test data). Both
  attempts are documented in the Model Information page in-app and
  in project history - this is a stated negative result, not a
  hidden failure.
- **Hybrid score weights (40% lexical / 60% semantic) are a reasoned**
  **starting point**, not weights tuned against a labeled validation
  dataset - M12's validation dataset was scaled down given project
  constraints (no classmates available to contribute handwriting
  samples).
- **M9 preprocessing steps (contrast enhancement, denoising,**
  **binarization) were tested and found to often hurt TrOCR accuracy**
  rather than help; the default pipeline uses only grayscale +
  deskew, based on that evidence. See pipeline_b_handwritten/preprocess_image.py
  for details.
