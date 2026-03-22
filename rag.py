import os
import anthropic
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

# En Streamlit Cloud, las variables vienen de st.secrets
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

# ── Configuración global ─────────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"  # multilingüe, mejor para español
CHUNK_SIZE   = 1200
CHUNK_OVERLAP= 200
RELEVANCE_THRESHOLD = 0.75  # mínimo de similitud coseno para usar un chunk

client = anthropic.Anthropic()
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)


# ── 1. INGESTA ────────────────────────────────────────────────────────────────

def load_and_split_pdf(pdf_path: str) -> list:
    """Carga un PDF y lo divide en chunks con overlap."""
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    # Agrega metadata útil a cada chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]  = i
        chunk.metadata["source"]    = Path(pdf_path).name
        chunk.metadata["char_count"]= len(chunk.page_content)

    return chunks


def ingest_document(pdf_path: str, collection_name: str = "documents") -> dict:
    """Pipeline completo: carga → split → embed → guarda en ChromaDB."""
    chunks = load_and_split_pdf(pdf_path)

    # Crea o añade a la colección existente (cosine para mejor relevancia)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectorstore.add_documents(chunks)

    return {
        "filename": Path(pdf_path).name,
        "pages":    len(set(c.metadata.get("page", 0) for c in chunks)),
        "chunks":   len(chunks),
    }


def get_ingested_docs(collection_name: str = "documents") -> list[str]:
    """Devuelve la lista de documentos ya procesados."""
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        docs = vectorstore.get()
        sources = list(set(
            m.get("source", "Desconocido")
            for m in docs.get("metadatas", [])
        ))
        return sources
    except Exception:
        return []


def delete_collection(collection_name: str = "documents"):
    """Borra todos los documentos de la colección."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    vectorstore.delete_collection()


# ── 2. RETRIEVAL ──────────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = 6,
                     collection_name: str = "documents") -> list:
    """Busca los k chunks más relevantes para la pregunta, filtrando por umbral."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Con distancia coseno: score 0 = idéntico, 1 = opuesto
    results = vectorstore.similarity_search_with_score(query, k=k)
    # Filtrar chunks con baja relevancia (convertir distancia a similitud)
    filtered = [
        (doc, score) for doc, score in results
        if (1 - score) >= RELEVANCE_THRESHOLD
    ]
    # Si ninguno pasa el umbral, devolver los 2 mejores de todas formas
    if not filtered and results:
        filtered = results[:2]
    return filtered


# ── 3. GENERACIÓN ─────────────────────────────────────────────────────────────

def build_prompt(query: str, context_docs: list) -> tuple[str, list]:
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

    # Formatea el contexto con metadata visible
    context_parts = []
    sources_used  = []

    for doc, score in context_docs:
        source = doc.metadata.get("source", "Desconocido")
        page   = doc.metadata.get("page", "?")
        relevance = round(max(0, (1 - score)) * 100, 1)

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


def ask(query: str, chat_history: list = None,
        collection_name: str = "documents") -> dict:
    """Función principal: pregunta → contexto → respuesta con fuentes."""

    # 1. Recuperar contexto relevante
    context_docs = retrieve_context(query, k=6, collection_name=collection_name)

    if not context_docs:
        return {
            "answer":  "No encontré documentos procesados. Sube un PDF primero.",
            "sources": [],
            "context_used": [],
        }

    # 2. Construir mensajes
    system_prompt, user_message, sources = build_prompt(query, context_docs)

    # 3. Construir historial de conversación para memoria
    messages = []
    if chat_history:
        for msg in chat_history[-6:]:  # últimos 3 intercambios
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # 4. Llamar a Claude
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system_prompt,
        messages=messages,
    )

    return {
        "answer":       response.content[0].text,
        "sources":      sources,
        "context_used": context_docs,
    }