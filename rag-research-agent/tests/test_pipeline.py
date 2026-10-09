"""Run from the project root:  python -m pytest -v

Fast tests use fake embeddings and a fake LLM (no downloads, no network).
Set RUN_SLOW_TESTS=1 to also test the real embedding model.
"""
import dataclasses
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docx import Document as DocxDocument  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from src.chunking import chunk_documents  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.graph import build_ingestion_graph, build_research_graph, run_ingestion, run_research  # noqa: E402
from src.llm import get_llm, translate_llm_error  # noqa: E402
from src.loaders import load_file  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.utils import (  # noqa: E402
    ConfigError,
    DocumentLoadError,
    EmptyDocumentError,
    EmptyQueryError,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    clean_text,
    cosine_similarity,
)
from src.vectorstore import VectorStoreManager  # noqa: E402

TEXT = (
    "Photosynthesis converts light energy into chemical energy in plants.\n\n"
    "The Krebs cycle is part of cellular respiration in mitochondria.\n\n"
    "Transformers use self-attention to model long-range dependencies."
)


# ---------------------------------------------------------------- helpers
def make_pdf(path: Path, text: str) -> None:
    """Write a minimal valid one-page PDF (no extra libraries needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)


@pytest.fixture
def settings(tmp_path):
    return dataclasses.replace(
        load_settings(),
        vector_db_path=tmp_path / "vs",
        documents_dir=tmp_path / "docs",
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        min_similarity=0.0,
        llm_provider="ollama",
    )


@pytest.fixture
def store(settings):
    return VectorStoreManager(settings.vector_db_path, "test_collection", DeterministicFakeEmbedding(size=64))


@pytest.fixture
def txt_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


# ------------------------------------------------------- document loading
def test_load_txt(txt_file):
    docs = load_file(txt_file)
    assert docs[0].metadata["source"] == "notes.txt"
    assert "Photosynthesis" in docs[0].page_content


def test_load_pdf_has_page_number(tmp_path):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, "Hello research world")
    docs = load_file(pdf)
    assert docs[0].metadata["page"] == 1
    assert "Hello" in docs[0].page_content


def test_load_docx(tmp_path):
    path = tmp_path / "report.docx"
    document = DocxDocument()
    document.add_paragraph("Quantum computing uses qubits.")
    document.save(str(path))
    assert "qubits" in load_file(path)[0].page_content


def test_invalid_and_empty_files(tmp_path):
    bad = tmp_path / "image.png"
    bad.write_bytes(b"123")
    with pytest.raises(DocumentLoadError):
        load_file(bad)
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        load_file(empty)


def test_clean_text():
    assert clean_text("infor-\nmation  is\nkey\n\n\n\nnext") == "information is key\n\nnext"


# ---------------------------------------------------------------- chunking
def test_chunking_metadata_and_validation(txt_file):
    docs = load_file(txt_file)
    chunks = chunk_documents(docs, 100, 20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    assert chunks[0].metadata["source"] == "notes.txt"
    with pytest.raises(ConfigError):
        chunk_documents(docs, 100, 100)


# -------------------------------------------------------------- embeddings
def test_embedding_dimension_fake():
    assert len(DeterministicFakeEmbedding(size=64).embed_query("hello")) == 64


@pytest.mark.skipif(not os.getenv("RUN_SLOW_TESTS"), reason="set RUN_SLOW_TESTS=1 (downloads model)")
def test_real_embedding_semantics():
    from src.embeddings import build_embeddings, get_embedding_dimension

    emb = build_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    assert get_embedding_dimension(emb) == 384
    cat, kitten, stock = emb.embed_documents(
        ["A cat sat on the mat.", "A kitten rests on a rug.", "Stock markets fell sharply today."]
    )
    assert cosine_similarity(cat, kitten) > cosine_similarity(cat, stock)


# --------------------------------------------------------------- retrieval
def test_vector_retrieval_and_persistence(settings, store, txt_file):
    chunks = chunk_documents(load_file(txt_file), 100, 20)
    store.add_chunks(chunks)
    n = store.count()
    assert n == len(chunks)

    retriever = Retriever(store, top_k=2, min_similarity=0.0)
    target = chunks[1].page_content
    assert retriever.retrieve(target)[0].document.page_content == target

    store.add_chunks(chunks)  # re-index: no duplicates
    assert store.count() == n

    reopened = VectorStoreManager(settings.vector_db_path, "test_collection", DeterministicFakeEmbedding(size=64))
    assert reopened.count() == n  # persisted


def test_empty_query_raises(store):
    with pytest.raises(EmptyQueryError):
        Retriever(store, 3, 0.0).retrieve("   ")


# ------------------------------------------------------------ LLM selection
def test_llm_selection(settings):
    assert type(get_llm(settings)).__name__ == "ChatOllama"
    gemini = dataclasses.replace(settings, llm_provider="gemini", gemini_api_key="")
    with pytest.raises(LLMAuthError):
        get_llm(gemini)
    gemini_keyed = dataclasses.replace(gemini, gemini_api_key="fake-key-for-test")
    assert type(get_llm(gemini_keyed)).__name__ == "ChatGoogleGenerativeAI"
    with pytest.raises(ConfigError):
        get_llm(dataclasses.replace(settings, llm_provider="nope"))


def test_error_translation(settings):
    assert isinstance(translate_llm_error(ConnectionError("Failed to connect to Ollama"), settings), LLMConnectionError)
    gem = dataclasses.replace(settings, llm_provider="gemini")
    assert isinstance(translate_llm_error(Exception("429 RESOURCE_EXHAUSTED"), gem), LLMRateLimitError)
    assert isinstance(translate_llm_error(Exception("API key not valid"), gem), LLMAuthError)


# ------------------------------------------------------------ full pipeline
def test_full_rag_pipeline(settings, store, txt_file):
    result = run_ingestion(build_ingestion_graph(settings, store), [txt_file])
    assert result["indexed_chunks"] > 0

    fake_llm = FakeListChatModel(
        responses=["### Answer\nPlants convert light to chemical energy [Source 1].",
                   "## 1. Executive Summary\nPlants use light [Source 1]."]
    )
    graph = build_research_graph(Retriever(store, 3, 0.0), fake_llm, settings)
    out = run_research(graph, "How do plants use light?", want_report=True)
    assert "chemical energy" in out["answer"]
    assert out["report"].startswith("# Research Report")
    assert "## 8. Sources" in out["report"]
    assert out["sources"] and out["sources"][0]["filename"] == "notes.txt"

    answer_only = run_research(
        build_research_graph(Retriever(store, 3, 0.0), FakeListChatModel(responses=["ok"]), settings),
        "anything", want_report=False,
    )
    assert answer_only.get("report", "") == ""


def test_no_relevant_context_skips_llm(settings, store, txt_file):
    run_ingestion(build_ingestion_graph(settings, store), [txt_file])
    llm = FakeListChatModel(responses=["SHOULD NOT BE USED"])
    graph = build_research_graph(Retriever(store, 3, min_similarity=1.01), llm, settings)
    out = run_research(graph, "unrelated question")
    assert out["insufficient_context"] is True
    assert "SHOULD NOT BE USED" not in out["answer"]
