import streamlit as st
import tempfile
import os
import logging

import anthropic
from langchain_huggingface import HuggingFaceEmbeddings

from rag import (
    ingest_document, get_ingested_docs, ask, delete_collection,
    set_embeddings, set_client, create_embeddings, create_client,
    MAX_UPLOAD_SIZE_MB, MAX_DOCUMENTS, EMBED_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat con tus documentos",
    page_icon="📄",
    layout="wide",
)

# ── Streamlit secrets -> env (solo en Streamlit Cloud) ───────────────────────
if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]


# ── Cache de recursos pesados ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def _cached_embeddings() -> HuggingFaceEmbeddings:
    return create_embeddings()


@st.cache_resource(show_spinner=False)
def _cached_client() -> anthropic.Anthropic:
    return create_client()


# Inyectar recursos cacheados en rag.py
set_embeddings(_cached_embeddings())
set_client(_cached_client())

# ── Estado inicial ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Documentos")

    uploaded = st.file_uploader(
        "Sube uno o varios PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        active_docs = get_ingested_docs()

        for file in uploaded:
            # Validacion: tamanio
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            if file_size_mb > MAX_UPLOAD_SIZE_MB:
                st.error(f"{file.name} excede el limite de {MAX_UPLOAD_SIZE_MB} MB ({file_size_mb:.1f} MB)")
                continue

            # Validacion: limite de documentos
            if len(active_docs) >= MAX_DOCUMENTS:
                st.error(f"Limite de {MAX_DOCUMENTS} documentos alcanzado. Elimina algunos primero.")
                break

            if file.name in active_docs:
                st.info(f"{file.name} ya esta cargado")
                continue

            with st.spinner(f"Procesando {file.name}..."):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    result = ingest_document(tmp_path, original_name=file.name)
                    active_docs.append(file.name)

                    st.success(
                        f"✓ {result['filename']} — "
                        f"{result['pages']} pags, {result['chunks']} fragmentos"
                    )
                except Exception as e:
                    st.error(f"Error al procesar {file.name}: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    # Documentos activos
    st.divider()
    st.subheader("Documentos activos")
    docs = get_ingested_docs()

    if docs:
        for doc in docs:
            st.markdown(f"- {doc}")

        if st.button("Limpiar todos los documentos", type="secondary"):
            delete_collection()
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("Ninguno todavia — sube un PDF arriba")

    st.divider()
    st.caption("Consejos para mejores respuestas:")
    st.caption("• Se especifico en tus preguntas")
    st.caption("• Menciona el documento si tienes varios")
    st.caption("• Pide citas textuales si las necesitas")

# ── Main: Chat ────────────────────────────────────────────────────────────────
st.title("💬 Chat con tus documentos")
st.caption("Powered by Claude + RAG · Las respuestas siempre citan la fuente")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Fuentes usadas ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['source']}** — Pagina {s['page']} "
                        f"· Relevancia: {s['relevance']}%"
                    )
                    st.caption(s["preview"])

if prompt := st.chat_input("Hazle una pregunta a tus documentos..."):

    if not get_ingested_docs():
        st.warning("Sube al menos un PDF antes de hacer preguntas.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en tus documentos..."):
            result = ask(
                query=prompt,
                chat_history=st.session_state.messages[:-1],
            )

        st.write(result["answer"])

        if result["sources"]:
            with st.expander(f"Fuentes usadas ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(
                        f"**{s['source']}** — Pagina {s['page']} "
                        f"· Relevancia: {s['relevance']}%"
                    )
                    st.caption(s["preview"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
