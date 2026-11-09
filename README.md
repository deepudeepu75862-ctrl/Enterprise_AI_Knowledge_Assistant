# Enterprise AI Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents, ask questions, retrieve semantically relevant passages, and view document/page references.

## Features

- Upload and process multiple PDF files
- Extract text page by page
- Split text into overlapping chunks
- Create semantic embeddings with `all-MiniLM-L6-v2`
- Retrieve relevant passages using cosine similarity
- Show source filename, page number, and similarity score
- Optional generated answers using Groq (Llama 3.3)
- Retrieval-only fallback mode when no API key is configured

## Architecture

```text
PDF Upload
   |
   v
PDF Text Extraction (pypdf)
   |
   v
Text Chunking (overlap)
   |
   v
Embedding Model (Sentence Transformers)
   |
   v
Vector Similarity Search (cosine similarity)
   |
   v
Top-K Relevant Passages
   |
   +------> Retrieval-only answer
   |
   +------> Optional Groq LLM answer with citations
```

## Requirements

- Python 3.10–3.12
- Internet connection for the first installation and model download
- Optional Groq API key for generated answers

## Installation (Windows PowerShell)

```powershell
cd Enterprise_AI_Knowledge_Assistant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
streamlit run app.py
```

Streamlit will show a local address, usually `http://localhost:8501`. Open it in your browser.

## Usage

1. Upload one or more text-based PDF files.
2. Click **Process documents**.
3. Enter a question and click **Ask AI**.
4. Expand the source references to inspect the retrieved passages and page numbers.

The embedding model downloads the first time the app runs. Scanned/image-only PDFs need OCR before upload.

## Enable generated answers (optional)

1. Create an API key in the Groq developer console.
2. In the app sidebar, enable **Use Groq LLM for generated answers**.
3. Paste your API key into the password field.
4. Ask a question.

Without Groq, the app still performs semantic retrieval and displays relevant document passages. Do not commit API keys to GitHub.

## Project structure

```text
Enterprise_AI_Knowledge_Assistant/
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Notes and limitations

- This starter project uses in-memory vectors; uploaded files are not persisted after the app restarts.
- PDF text extraction may not work on scanned PDFs.
- Similarity scores are retrieval signals, not guaranteed measures of answer correctness.
- For production use, add authentication, persistent vector storage, file-size limits, logging, access controls, and evaluation tests.

## GitHub

```powershell
git init
git add .
git commit -m "Build Enterprise AI Knowledge Assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Enterprise_AI_Knowledge_Assistant.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If `origin` already exists, use `git remote -v` to inspect it and `git remote set-url origin YOUR_REPOSITORY_URL` to update it.
