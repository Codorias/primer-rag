# DocChat — RAG-powered PDF Assistant

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-blueviolet?logo=anthropic&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green?logo=chainlink&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A conversational AI assistant that answers questions grounded strictly in your own PDF documents. Upload one or more files, ask anything in any language, and get answers with exact page-level citations, a confidence score, and full conversation history — no hallucinations, no guessing.

---

## How it works

```
PDF Upload
    │
    ▼
PyPDF loader → RecursiveTextSplitter (800 chars / 150 overlap)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2, runs locally)
    │
    ▼
ChromaDB  ◄──────────────────────────────────────────────┐
    │                                                     │
    │  Semantic search (top-4 chunks)                     │ persist
    ▼                                                     │
Context + last 6 messages ──► Claude Sonnet 4.6           │
    │                                                     │
    ▼                                                     │
Answer + [Source: file, p.X] citations + confidence score │
    │
    ▼
SQLite (chat_history.db) — persistent conversation log
```

Embeddings are computed locally (no API cost). Only the generation step calls the Anthropic API.

---

## Features

- **Persistent chat history** — conversations survive page refreshes and server restarts, stored in a local SQLite database
- **Conversation management** — start new chats, delete individual conversations, or clear all history from the sidebar
- **Confidence indicator** — each answer shows a color-coded score (Alta / Media / Baja) derived from the semantic similarity of the retrieved chunks
- **Source attribution** — every answer includes file name, page number, and relevance percentage
- **Multi-document support** — ingest and query across multiple PDFs simultaneously
- **Conversation memory** — retains the last 3 exchanges for coherent follow-up questions
- **Zero hallucination policy** — the model is instructed to explicitly state when information is not found in the documents
- **Cached vectorstore** — the ChromaDB instance is reused across queries within the same session

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Orchestration | LangChain + LangChain Community |
| Vector store | ChromaDB (local) |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (local) |
| PDF parsing | PyPDF |
| Chat persistence | SQLite (stdlib) |

---

## Getting started

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/Codorias/primer-rag.git
cd primer-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file at the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`. On first run, the embedding model (~90 MB) will be downloaded automatically.

---

## Deploy to Streamlit Cloud

1. Push the repository to GitHub (`chroma_db/`, `chat_history.db`, and `.env` are gitignored).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. In **App settings → Secrets**, add:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

4. Deploy. The first boot downloads the embedding model and may take a minute.

> **Note:** Streamlit Cloud has an ephemeral filesystem. The vector store and chat history are recreated on each cold start — users will need to re-upload documents after the app restarts. For fully persistent cloud storage, a hosted database (e.g. PostgreSQL) and vector store (e.g. Pinecone) would be needed.

---

## Project structure

```
primer-rag/
├── app.py              # Streamlit UI and session management
├── rag.py              # RAG pipeline: ingest / retrieve / generate
├── db.py               # SQLite layer for persistent chat history
├── requirements.txt
└── .streamlit/
    └── secrets.toml    # Local secrets (gitignored)
```

---

## License

MIT
