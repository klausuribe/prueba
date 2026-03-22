# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) application using Python 3.12. The project is in early stages with `app.py` and `rag.py` as the main source files.

## Tech Stack

- **LLM**: Anthropic Claude via `langchain-anthropic`
- **Framework**: LangChain + LangGraph for orchestration
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Document Loading**: PyPDF
- **UI**: Streamlit
- **Environment**: Python 3.12 venv, environment variables via `python-dotenv`

## Setup & Running

```bash
source venv/bin/activate
streamlit run app.py
```

## Environment

API keys and configuration are loaded from `.env` via `python-dotenv`. The `.env` file must be present but is not committed to version control.
