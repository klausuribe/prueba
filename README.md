# Chat con tus Documentos

RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents and have intelligent conversations about their content, with source citations and relevance scoring.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude_Sonnet_4.6-191919?logo=anthropic&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F61)

## Features

- **PDF Document Ingestion** — Upload one or multiple PDFs; documents are automatically chunked, embedded, and stored in a vector database
- **Semantic Search** — Cosine similarity retrieval with multilingual embeddings, optimized for Spanish and English
- **Source Citations** — Every answer includes the source document, page number, and relevance percentage
- **Conversation Memory** — Maintains context across exchanges for follow-up questions
- **Relevance Filtering** — Only includes document fragments above a similarity threshold, with a fallback mechanism
- **Duplicate Detection** — Prevents re-ingestion of already processed documents

## Architecture

```
User (PDF / Question)
       │
       ▼
  Streamlit UI (app.py)
       │
       ├── Upload ─► load_and_split_pdf() ─► ChromaDB
       │
       └── Query  ─► retrieve_context()    ─► Cosine similarity search
                     build_prompt()         ─► Context + instructions
                     Claude Sonnet 4.6      ─► Answer with citations
                            │
                            ▼
                  Response + Sources
```

## Tech Stack

| Component        | Technology                                    |
| ---------------- | --------------------------------------------- |
| **Frontend**     | Streamlit                                     |
| **LLM**          | Anthropic Claude Sonnet 4.6                   |
| **Embeddings**   | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| **Vector Store** | ChromaDB (cosine distance)                    |
| **Orchestration**| LangChain                                     |
| **PDF Parsing**  | PyPDF                                         |

## Getting Started

### Prerequisites

- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/klausuribe/prueba.git
cd prueba
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your-api-key-here
```

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. **Upload PDFs** — Use the sidebar to drag and drop or browse for PDF files
2. **Ask questions** — Type your question in the chat input
3. **Review sources** — Expand the "Fuentes usadas" section to see which document fragments were used, with page numbers and relevance scores

## Project Structure

```
├── app.py               # Streamlit UI and chat interface
├── rag.py               # RAG pipeline (ingestion, retrieval, generation)
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── .gitignore
├── .streamlit/
│   └── config.toml      # Streamlit theme and upload settings
└── chroma_db/           # Vector database (generated at runtime)
```

## Deployment

The app is ready for [Streamlit Community Cloud](https://streamlit.io/cloud). Add your `ANTHROPIC_API_KEY` in the Streamlit secrets management dashboard.

## License

MIT
