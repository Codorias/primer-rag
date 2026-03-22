import os
import anthropic
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Configuración global ─────────────────────────────────────────────────────
CHROMA_PATH   = "chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

client     = anthropic.Anthropic()
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Caché del vectorstore — se reutiliza entre llamadas en la misma sesión
_store: Chroma | None = None


def _get_store(collection_name: str = "documents") -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
    return _store


# ── 1. INGESTA ────────────────────────────────────────────────────────────────

def load_and_split_pdf(pdf_path: str, display_name: str | None = None) -> list:
    """Carga un PDF y lo divide en chunks con overlap."""
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    source = display_name or Path(pdf_path).name
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]   = i
        chunk.metadata["source"]     = source
        chunk.metadata["char_count"] = len(chunk.page_content)

    return chunks


def ingest_document(pdf_path: str, filename: str | None = None,
                    collection_name: str = "documents") -> dict:
    """Pipeline completo: carga → split → embed → guarda en ChromaDB."""
    display_name = filename or Path(pdf_path).name
    chunks = load_and_split_pdf(pdf_path, display_name)
    _get_store(collection_name).add_documents(chunks)

    return {
        "filename": display_name,
        "pages":    len(set(c.metadata.get("page", 0) for c in chunks)),
        "chunks":   len(chunks),
    }


def get_ingested_docs(collection_name: str = "documents") -> list[str]:
    """Devuelve la lista de documentos ya procesados."""
    try:
        docs = _get_store(collection_name).get()
        return list(set(
            m.get("source", "Desconocido")
            for m in docs.get("metadatas", [])
        ))
    except Exception:
        return []


def delete_collection(collection_name: str = "documents") -> None:
    """Borra todos los documentos de la colección e invalida el caché."""
    global _store
    _get_store(collection_name).delete_collection()
    _store = None


# ── 2. RETRIEVAL ──────────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = 4,
                     collection_name: str = "documents") -> list:
    """Busca los k chunks más relevantes para la pregunta."""
    # similarity_search_with_score devuelve (doc, score) — score más bajo = más similar
    return _get_store(collection_name).similarity_search_with_score(query, k=k)


# ── 3. GENERACIÓN ─────────────────────────────────────────────────────────────

def build_prompt(query: str, context_docs: list) -> tuple[str, str, list]:
    """Construye el system prompt y el contexto a partir de los chunks."""

    system_prompt = """Eres un asistente experto que responde preguntas basándose \
ÚNICAMENTE en el contexto proporcionado de los documentos.

REGLAS:
1. Responde solo con información del contexto. Si no está, di claramente que no \
lo encontraste en los documentos.
2. Al final de cada dato importante, cita la fuente así: [Fuente: nombre_archivo, p.X]
3. Sé preciso y conciso. No inventes ni supongas.
4. Responde en el mismo idioma que la pregunta.
5. Si la pregunta no puede responderse con el contexto, sugiere qué documento \
podría tener esa información."""

    context_parts = []
    sources_used  = []

    for doc, score in context_docs:
        source    = doc.metadata.get("source", "Desconocido")
        page      = doc.metadata.get("page", "?")
        relevance = round((1 - score) * 100, 1)

        context_parts.append(
            f"[Fragmento de: {source}, página {page}, "
            f"relevancia: {relevance}%]\n{doc.page_content}"
        )
        sources_used.append({
            "source":    source,
            "page":      page,
            "relevance": relevance,
            "preview":   doc.page_content[:150] + "...",
        })

    context_text = "\n\n---\n\n".join(context_parts)

    user_message = f"""Contexto de los documentos:

{context_text}

---

Pregunta: {query}"""

    return system_prompt, user_message, sources_used


def ask(query: str, chat_history: list | None = None,
        collection_name: str = "documents") -> dict:
    """Función principal: pregunta → contexto → respuesta con fuentes."""

    context_docs = retrieve_context(query, k=4, collection_name=collection_name)

    if not context_docs:
        return {
            "answer":       "No encontré documentos procesados. Sube un PDF primero.",
            "sources":      [],
            "context_used": [],
        }

    system_prompt, user_message, sources = build_prompt(query, context_docs)

    messages = []
    if chat_history:
        for msg in chat_history[-6:]:  # últimos 3 intercambios
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=system_prompt,
            messages=messages,
        )
        answer = response.content[0].text
    except anthropic.APIError as e:
        answer = f"Error al contactar la API de Claude: {e}"
        sources = []

    confidence = (
        round(sum(s["relevance"] for s in sources) / len(sources), 1)
        if sources else 0.0
    )

    return {
        "answer":       answer,
        "sources":      sources,
        "context_used": context_docs,
        "confidence":   confidence,
    }
