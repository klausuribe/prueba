import streamlit as st
import tempfile
import os
from rag import ingest_document, get_ingested_docs, ask, delete_collection

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat con tus documentos",
    page_icon="📄",
    layout="wide",
)

# ── Estado inicial ────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "docs_loaded" not in st.session_state: st.session_state.docs_loaded = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Documentos")

    # Upload
    uploaded = st.file_uploader(
        "Sube uno o varios PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        for file in uploaded:
            already = get_ingested_docs()
            if file.name not in already:
                with st.spinner(f"Procesando {file.name}..."):
                    # Guardar temp y procesar
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    result = ingest_document(tmp_path)
                    os.unlink(tmp_path)

                    st.success(
                        f"✓ {result['filename']} — "
                        f"{result['pages']} págs, {result['chunks']} fragmentos"
                    )
            else:
                st.info(f"{file.name} ya está cargado")

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
        st.caption("Ninguno todavía — sube un PDF arriba")

    # Tips
    st.divider()
    st.caption("Consejos para mejores respuestas:")
    st.caption("• Sé específico en tus preguntas")
    st.caption("• Menciona el documento si tienes varios")
    st.caption("• Pide citas textuales si las necesitas")

# ── Main: Chat ────────────────────────────────────────────────────────────────
st.title("💬 Chat con tus documentos")
st.caption("Powered by Claude + RAG · Las respuestas siempre citan la fuente")

# Render historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Mostrar fuentes si las hay
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

    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en tus documentos..."):
            result = ask(
                query=prompt,
                chat_history=st.session_state.messages[:-1],
            )

        st.write(result["answer"])

        # Fuentes
        if result["sources"]:
            with st.expander(f"Fuentes usadas ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(
                        f"**{s['source']}** — Página {s['page']} "
                        f"· Relevancia: {s['relevance']}%"
                    )
                    st.caption(s["preview"])

    # Guardar en historial
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })