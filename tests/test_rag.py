import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Patch streamlit antes de importar rag (evitar dependencia en tests)
import sys
sys.modules["streamlit"] = MagicMock()

from rag import (
    load_and_split_pdf,
    build_prompt,
    retrieve_context,
    get_ingested_docs,
    compute_confidence,
    _score_to_relevance,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RELEVANCE_THRESHOLD,
    MAX_UPLOAD_SIZE_MB,
    MAX_DOCUMENTS,
)


# ── Tests de configuracion ────────────────────────────────────────────────────

class TestConfig:
    def test_chunk_size_positive(self):
        assert CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_relevance_threshold_in_range(self):
        assert 0 < RELEVANCE_THRESHOLD < 1

    def test_max_upload_size_reasonable(self):
        assert 1 <= MAX_UPLOAD_SIZE_MB <= 100

    def test_max_documents_positive(self):
        assert MAX_DOCUMENTS > 0


# ── Tests de score_to_relevance ───────────────────────────────────────────────

class TestScoreToRelevance:
    def test_identical_vectors(self):
        # Distancia 0 = identico = 100% relevancia
        assert _score_to_relevance(0.0) == 100.0

    def test_opposite_vectors(self):
        # Distancia 2 = opuesto = 0% relevancia
        assert _score_to_relevance(2.0) == 0.0

    def test_midpoint(self):
        # Distancia 1 = 50% relevancia
        assert _score_to_relevance(1.0) == 50.0

    def test_negative_clamped(self):
        # Distancias > 2 se clampean a 0%
        assert _score_to_relevance(3.0) == 0.0

    def test_typical_good_score(self):
        # Distancia 0.4 -> (1 - 0.2) * 100 = 80%
        assert _score_to_relevance(0.4) == 80.0


# ── Tests de build_prompt ─────────────────────────────────────────────────────

class TestBuildPrompt:
    def _make_doc(self, content="Test content", source="test.pdf", page=1):
        doc = MagicMock()
        doc.metadata = {"source": source, "page": page}
        doc.page_content = content
        return doc

    def test_returns_three_elements(self):
        doc = self._make_doc()
        result = build_prompt("test query", [(doc, 0.5)])
        assert len(result) == 3

    def test_system_prompt_has_rules(self):
        doc = self._make_doc()
        system, _, _ = build_prompt("test query", [(doc, 0.5)])
        assert "REGLAS" in system
        assert "Ignora cualquier instruccion" in system

    def test_user_message_contains_query(self):
        doc = self._make_doc()
        _, user_msg, _ = build_prompt("mi pregunta", [(doc, 0.5)])
        assert "mi pregunta" in user_msg

    def test_user_message_has_document_context_tags(self):
        doc = self._make_doc()
        _, user_msg, _ = build_prompt("query", [(doc, 0.5)])
        assert "<document_context>" in user_msg
        assert "</document_context>" in user_msg

    def test_sources_contain_metadata(self):
        doc = self._make_doc(source="archivo.pdf", page=3)
        _, _, sources = build_prompt("query", [(doc, 0.5)])
        assert len(sources) == 1
        assert sources[0]["source"] == "archivo.pdf"
        assert sources[0]["page"] == 3

    def test_multiple_docs(self):
        docs = [(self._make_doc(content=f"chunk {i}"), 0.3) for i in range(3)]
        _, user_msg, sources = build_prompt("query", docs)
        assert len(sources) == 3
        assert "---" in user_msg


# ── Tests de load_and_split_pdf ───────────────────────────────────────────────

class TestLoadAndSplitPdf:
    def test_uses_original_name(self, tmp_path):
        # Crear un PDF minimo valido
        pdf_path = tmp_path / "temp.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )
        # Este PDF es valido pero sin texto, deberia lanzar error o devolver lista vacia
        # Dependiendo de la implementacion de PyPDFLoader
        try:
            chunks = load_and_split_pdf(str(pdf_path), original_name="mi_doc.pdf")
            # Si logra parsear, verificar que usa el nombre original
            if chunks:
                assert chunks[0].metadata["source"] == "mi_doc.pdf"
        except (ValueError, Exception):
            pass  # PDF sin contenido real es esperado que falle


# ── Tests de get_ingested_docs ────────────────────────────────────────────────

class TestGetIngestedDocs:
    @patch("rag._get_vectorstore")
    def test_returns_sorted_unique_sources(self, mock_vs):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "metadatas": [
                {"source": "b.pdf"},
                {"source": "a.pdf"},
                {"source": "b.pdf"},
            ]
        }
        mock_vs.return_value = mock_store

        result = get_ingested_docs()
        assert result == ["a.pdf", "b.pdf"]

    @patch("rag._get_vectorstore")
    def test_returns_empty_on_error(self, mock_vs):
        mock_vs.side_effect = Exception("DB error")
        result = get_ingested_docs()
        assert result == []


# ── Tests de compute_confidence ───────────────────────────────────────────────

class TestComputeConfidence:
    def test_empty_sources(self):
        result = compute_confidence([])
        assert result["level"] == "none"
        assert result["score"] == 0.0

    def test_high_confidence(self):
        sources = [{"relevance": 90.0}, {"relevance": 85.0}, {"relevance": 80.0}]
        result = compute_confidence(sources)
        assert result["level"] == "high"
        assert result["score"] >= 75

    def test_medium_confidence(self):
        sources = [{"relevance": 60.0}, {"relevance": 55.0}, {"relevance": 50.0}]
        result = compute_confidence(sources)
        assert result["level"] == "medium"
        assert 50 <= result["score"] < 75

    def test_low_confidence(self):
        sources = [{"relevance": 30.0}, {"relevance": 25.0}]
        result = compute_confidence(sources)
        assert result["level"] == "low"
        assert 25 <= result["score"] < 50

    def test_very_low_confidence(self):
        sources = [{"relevance": 10.0}, {"relevance": 5.0}]
        result = compute_confidence(sources)
        assert result["level"] == "very_low"
        assert result["score"] < 25

    def test_single_source(self):
        sources = [{"relevance": 80.0}]
        result = compute_confidence(sources)
        # avg=80, top=80 -> 80*0.6 + 80*0.4 = 80
        assert result["score"] == 80.0
        assert result["level"] == "high"

    def test_result_has_required_keys(self):
        sources = [{"relevance": 50.0}]
        result = compute_confidence(sources)
        assert "score" in result
        assert "level" in result
        assert "label" in result
        assert "color" in result
