# Multi-Agent AI Research Assistant

An academic prototype for analysing uploaded scientific papers using semantic retrieval and a six-agent Gemini workflow.

## Features
- Upload multiple text-based research PDFs.
- Extract page-level text using PyMuPDF.
- Split text into overlapping chunks.
- Create sentence-transformer embeddings.
- Retrieve semantically relevant evidence.
- Run six specialised Gemini-powered agents:
  1. Research Discovery
  2. Paper Analysis
  3. Comparative Analysis
  4. Research Gap Identification
  5. Fact-Checking
  6. Report Generation
- Download the final report as Markdown.
- Display evidence source filenames and page numbers.

## Important scope and limitations
This is a working prototype, not a validated clinical or scientific decision system. Retrieval similarity is not proof of relevance or truth. The fact-checking agent checks claims only against the uploaded/retrieved passages; it does not independently verify the entire scientific literature. Scanned image-only PDFs are not OCR-processed. The uploaded files and generated embeddings are held in app memory during the session; this prototype does not provide persistent user storage, authentication, or a database.

The current version uses six sequential specialised LLM calls coordinated by a Python workflow. It does not yet implement LangGraph, model fine-tuning, or formal benchmark evaluation. These are planned research extensions and must not be reported as already completed.

## Requirements
- Python 3.10–3.12 recommended
- Gemini API key from https://aistudio.google.com/apikey
- Internet connection for model and embedding downloads

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit (usually http://localhost:8501). Enter your Gemini API key in the sidebar. The key is used for the current session and is not stored by the app.

## First test
1. Upload 2–5 text-based research PDFs.
2. Click **Process uploaded papers**.
3. Enter a focused research question.
4. Expand **Preview retrieved evidence** to inspect retrieval.
5. Enter your Gemini API key and click **Run six-agent research analysis**.
6. Inspect each agent output and download the Markdown report.
7. Check every claim against the original PDFs before academic use.

## Deploy on Streamlit Community Cloud
1. Push this project to a GitHub repository.
2. Open https://share.streamlit.io/ and connect GitHub.
3. Create an app using this repository, branch `main`, and entrypoint `app.py`.
4. In Advanced settings / Secrets, add:
```toml
GEMINI_API_KEY = "your-real-key"
GEMINI_MODEL = "gemini-2.5-flash"
```
5. Deploy. Do not put API keys in source code or commit them to GitHub.

## Dataset strategy
For a stable first prototype, users upload their own research PDFs. SciFact and QASPER can be integrated in a later evaluation module. Do not upload copyrighted papers to a public repository unless their licences permit redistribution.
