import streamlit as st
import tempfile
import os

from rag import ingest_document, get_ingested_docs, ask, delete_collection, embeddings, client
from db import (
    init_db, create_conversation, add_message,
    get_conversations, get_messages, delete_conversation, clear_all_conversations,
)

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat con tus documentos",
    page_icon="📄",
    layout="wide",
)

init_db()


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def _load_resources():
    return embeddings, client


_load_resources()

# ── Estado inicial ────────────────────────────────────────────────────────────
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def _load_conversation(conv_id: int) -> None:
    st.session_state.current_conv_id = conv_id
    st.session_state.messages = get_messages(conv_id)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Historial ─────────────────────────────────────────────────────────────
    st.subheader("💬 Conversaciones")

    if st.button("＋ Nueva conversación", use_container_width=True):
        st.session_state.current_conv_id = None
        st.session_state.messages = []
        st.rerun()

    conversations = get_conversations()

    if conversations:
        for conv in conversations[:30]:
            is_active = st.session_state.current_conv_id == conv["id"]
            col_title, col_del = st.columns([5, 1])

            with col_title:
                title = f"**{conv['title']}**" if is_active else conv["title"]
                if st.button(title, key=f"conv_{conv['id']}", use_container_width=True):
                    _load_conversation(conv["id"])
                    st.rerun()

            with col_del:
                if st.button("🗑", key=f"del_{conv['id']}"):
                    delete_conversation(conv["id"])
                    if st.session_state.current_conv_id == conv["id"]:
                        st.session_state.current_conv_id = None
                        st.session_state.messages = []
                    st.rerun()

        st.divider()
        if st.button("🗑 Borrar todo el historial", use_container_width=True):
            clear_all_conversations()
            st.session_state.current_conv_id = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("Sin conversaciones aún")

    # ── Documentos ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Documentos")

    uploaded = st.file_uploader(
        "Sube uno o varios PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        for file in uploaded:
            if file.name in get_ingested_docs():
                st.info(f"{file.name} ya está cargado")
                continue

            with st.spinner(f"Procesando {file.name}..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    result = ingest_document(tmp_path, filename=file.name)
                    st.success(
                        f"✓ {result['filename']} — "
                        f"{result['pages']} págs, {result['chunks']} fragmentos"
                    )
                except Exception as e:
                    st.error(f"Error procesando {file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    st.divider()
    st.subheader("Documentos activos")
    docs = get_ingested_docs()

    if docs:
        for doc in docs:
            st.markdown(f"- {doc}")
        if st.button("Limpiar todos los documentos", type="secondary"):
            delete_collection()
            st.rerun()
    else:
        st.caption("Ninguno todavía — sube un PDF arriba")

    st.divider()
    st.caption("Consejos para mejores respuestas:")
    st.caption("• Sé específico en tus preguntas")
    st.caption("• Menciona el documento si tienes varios")
    st.caption("• Pide citas textuales si las necesitas")

# ── Main: Chat ────────────────────────────────────────────────────────────────
st.title("💬 Chat con tus documentos")
st.caption("Powered by Claude + RAG · Las respuestas siempre citan la fuente")

# Render historial de la conversación activa
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("confidence") is not None and msg["role"] == "assistant":
            conf = msg["confidence"]
            if conf >= 75:
                label, color = "Alta", "🟢"
            elif conf >= 50:
                label, color = "Media", "🟡"
            else:
                label, color = "Baja", "🔴"
            st.caption(f"{color} Confianza de la respuesta: **{label}** ({conf}%)")
            st.progress(max(0, min(100, int(conf))) / 100)
        if msg.get("sources"):
            with st.expander(f"Fuentes usadas ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['source']}** — Página {s['page']} "
                        f"· Relevancia: {s['relevance']}%"
                    )
                    st.caption(s["preview"])

# Input del usuario
if prompt := st.chat_input("Hazle una pregunta a tus documentos..."):

    if not get_ingested_docs():
        st.warning("Sube al menos un PDF antes de hacer preguntas.")
        st.stop()

    # Crear conversación si es la primera pregunta
    if st.session_state.current_conv_id is None:
        st.session_state.current_conv_id = create_conversation(prompt)

    conv_id = st.session_state.current_conv_id

    add_message(conv_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en tus documentos..."):
            try:
                result = ask(
                    query=prompt,
                    chat_history=st.session_state.messages[:-1],
                )
            except Exception as e:
                st.error(f"Error inesperado: {e}")
                st.stop()

        st.write(result["answer"])

        conf = result.get("confidence", 0.0)
        if conf >= 75:
            label, color = "Alta", "🟢"
        elif conf >= 50:
            label, color = "Media", "🟡"
        else:
            label, color = "Baja", "🔴"
        st.caption(f"{color} Confianza de la respuesta: **{label}** ({conf}%)")
        st.progress(max(0, min(100, int(conf))) / 100)

        if result["sources"]:
            with st.expander(f"Fuentes usadas ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(
                        f"**{s['source']}** — Página {s['page']} "
                        f"· Relevancia: {s['relevance']}%"
                    )
                    st.caption(s["preview"])

    add_message(conv_id, "assistant", result["answer"],
                sources=result["sources"], confidence=conf)
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    result["answer"],
        "sources":    result["sources"],
        "confidence": conf,
    })
