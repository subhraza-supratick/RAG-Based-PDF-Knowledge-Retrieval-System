# DOCSENSE Local AI — PDF Document Intelligence

A portfolio-grade **Retrieval-Augmented Generation (RAG)** web application for semantic PDF document intelligence. Built with **Django**, PyPDF, LangChain, Google Gemini, SQLite vector storage, and a Vanilla JavaScript SPA.

---

## Features

- **Page-aware PDF extraction** — PyPDF with blank/scanned page detection and financial table preservation
- **Recursive chunking** — LangChain `RecursiveCharacterTextSplitter` with exact page metadata per chunk
- **Gemini embeddings** — `text-embedding-004` (768-dim) stored as NumPy float32 BLOBs in SQLite
- **NumPy cosine similarity search** — Pure NumPy retriever with no external vector DB required
- **Grounded RAG answers** — Gemini `gemini-2.5-flash` LLM strictly grounded to retrieved context
- **Exact page citations** — Every answer sourced to specific page numbers with collapsible accordions
- **Benchmark suite** — 30-question evaluation comparing semantic vs keyword baseline retrieval
- **Full test suite** — pytest tests for extraction, chunking, DB, retriever, RAG, and API

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend Framework | Django 5.x + Django REST Framework |
| PDF Extraction | PyPDF |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Google Gemini `text-embedding-004` |
| LLM | Google Gemini `gemini-2.5-flash` |
| Vector Storage | SQLite (NumPy float32 BLOBs) |
| Similarity Search | NumPy cosine similarity |
| Frontend | Vanilla HTML + CSS + JavaScript SPA |
| Testing | pytest |
| Containerization | Docker |

---

## Project Structure

```
document-rag/
├── backend/
│   ├── config.py           # Settings, env loading, Gemini client setup
│   ├── database.py         # SQLite document & chunk persistence
│   ├── pdf_processor.py    # PyPDF extraction, cleaning, scanned detection
│   ├── chunker.py          # LangChain chunking with page metadata
│   ├── embeddings.py       # Gemini embedding generation + mock fallback
│   ├── retriever.py        # Semantic cosine + baseline keyword retrievers
│   ├── rag.py              # Gemini grounded QA with page citations
│   ├── views.py            # Django REST API views
│   └── urls.py             # API route definitions
├── frontend/
│   ├── index.html          # SPA — 3 states: Upload, Indexing, Chat
│   ├── style.css           # Dark mode glassmorphism theme
│   └── app.js              # Vanilla JS state machine & API calls
├── tests/
│   ├── test_pdf_processor.py
│   ├── test_chunker.py
│   ├── test_db.py
│   ├── test_retriever.py
│   ├── test_rag.py
│   └── test_api.py
├── benchmarks/
│   ├── benchmark.py        # Latency & Hit@5 evaluation script
│   └── dataset.json        # 30 analyst-style benchmark questions
├── data/
│   ├── rag.db              # SQLite vector database (auto-created)
│   └── uploads/            # Uploaded PDF files
├── document_rag/           # Django project settings
├── manage.py
├── requirements.txt
├── Dockerfile
└── .env
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd document-rag
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env .env.local
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_actual_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. Run the server

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000 in your browser.

### 4. Use the app

1. **Upload** a PDF (drag & drop or browse)
2. Watch the **indexing pipeline** progress (Extract → Chunk → Embed → Save)
3. **Ask questions** in the chat interface
4. View **grounded answers** with collapsible page citations

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health/` | Health check |
| POST | `/api/documents/upload/` | Upload a PDF file |
| POST | `/api/documents/<id>/index/` | Index uploaded document |
| GET | `/api/documents/` | List all indexed documents |
| DELETE | `/api/documents/<id>/` | Delete document and chunks |
| POST | `/api/ask/` | Ask a question (RAG query) |

### POST /api/ask/ — Request Body

```json
{
  "question": "What was the total revenue?",
  "document_id": 1,
  "top_k": 5
}
```

### POST /api/ask/ — Response

```json
{
  "answer": "Total revenue was $2.4B based on the annual report...",
  "sources": [
    {
      "chunk_id": 12,
      "page": 4,
      "filename": "annual_report.pdf",
      "text": "Revenue for FY2024 was $2.4 billion...",
      "score": 0.912
    }
  ],
  "metrics": {
    "retrieval_ms": 45.2,
    "generation_ms": 1230.5
  }
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Running Benchmarks

First index at least one document, then:

```bash
python benchmarks/benchmark.py --document_id 1
```

Output includes:
- Average & P95 retrieval latency (Semantic vs Baseline)
- Hit@5 rates
- Latency improvement percentage
- Saved JSON results in `benchmarks/benchmark_results.json`

---

## Docker

```bash
docker build -t docsense-local-ai .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key docsense-local-ai
```

---

## Notes

- **No Gemini API key?** The app still works using a deterministic mock embedding fallback — semantic similarity will be hash-based instead of LLM-based.
- **Scanned PDFs** are detected and rejected with a clear error message (OCR required).
- **SQLite** is used for zero-setup vector storage — no external vector DB needed.
