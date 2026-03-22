# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
source venv/bin/activate
streamlit run app.py
```

The app runs at `http://localhost:8501`. There are no tests.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for PDF documents with a Spanish-language UI.

**Two-file structure:**

- `rag.py` — core pipeline (ingest → retrieve → generate)
- `app.py` — Streamlit frontend

**Pipeline flow:**

1. **Ingest** (`rag.py:load_and_split_pdf`, `ingest_document`): PDF → 800-char chunks (150 overlap) → SentenceTransformer embeddings → ChromaDB
2. **Retrieve** (`rag.py:retrieve_context`): query → semantic search → top 4 chunks with page metadata
3. **Generate** (`rag.py:build_prompt`, `ask`): retrieved chunks + last 6 chat messages → Claude API → response with source citations

**Key configuration in `rag.py`:**
- `EMBED_MODEL = "all-MiniLM-L6-v2"` — local SentenceTransformer (no API cost)
- `CHROMA_PATH = "chroma_db"` — local vector store (gitignored)
- Claude model: `claude-sonnet-4-6`, max 1500 tokens
- Chunk size: 800 chars, overlap: 150 chars

**Dependencies:** `anthropic`, `langchain`, `langchain-community`, `chromadb`, `sentence-transformers`, `pypdf`, `streamlit`, `python-dotenv`

**Environment:** Requires `ANTHROPIC_API_KEY` in `.env`.
