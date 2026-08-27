# Handwritten Content Evaluator

A system for handwritten answer recognition, plagiarism detection, and rubric-based automated mark evaluation.

## Status

🚧 Early setup — Milestone 1 (dataset + handwriting recognition) in progress.

## Milestones

1. **Dataset + Handwriting Text Recognition (HTR)** — build a writer-independent handwritten dataset and fine-tune a pretrained HTR model, evaluated with CER/WER.
2. **Plagiarism Detection + Rubric-Based Mark Evaluation** — layered similarity detection (lexical → semantic) and a structured rubric evaluator with explicit evidence per criterion.
3. **Full System + Deployment** — FastAPI + PostgreSQL + React integration, tested and deployed on Render.

## Tech Stack

- **ML:** Python 3.12, PyTorch, Hugging Face Transformers, scikit-learn, sentence-transformers
- **Image processing:** OpenCV, Pillow, NumPy
- **Backend:** FastAPI, PostgreSQL (SQLAlchemy + Alembic)
- **Frontend:** React, Vite, TypeScript
- **Experiment tracking:** MLflow
- **Testing:** pytest
- **Deployment:** Docker, Render

## Setup

```
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(`requirements.txt` will be added as dependencies are introduced.)
