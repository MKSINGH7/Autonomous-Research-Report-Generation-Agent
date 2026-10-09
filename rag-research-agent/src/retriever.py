"""Semantic retrieval, context construction and source formatting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from src.utils import EmptyQueryError, NoDocumentsIndexedError
from src.vectorstore import VectorStoreManager


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus its relevance score (higher = more similar, 0..1)."""

    document: Document
    score: float

    @property
    def filename(self) -> str:
        return str(self.document.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        page = self.document.metadata.get("page")
        return int(page) if page is not None else None

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata.get("chunk_id", ""))


class Retriever:
    """Query -> embedding -> vector search -> top-K -> threshold filter."""

    def __init__(self, store: VectorStoreManager, top_k: int, min_similarity: float) -> None:
        self._store = store
        self._top_k = top_k
        self._min_similarity = min_similarity

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            raise EmptyQueryError("Please type a research question first.")
        if self._store.count() == 0:
            raise NoDocumentsIndexedError(
                "No documents are indexed yet. Upload files and click "
                "'Process / Index documents' first."
            )
        raw = self._store.search(query, k=self._top_k)
        results = [RetrievedChunk(doc, float(score)) for doc, score in raw]
        results = [r for r in results if r.score >= self._min_similarity]
        return sorted(results, key=lambda r: r.score, reverse=True)


def build_context(
    chunks: list[RetrievedChunk], max_chars: int
) -> tuple[str, list[RetrievedChunk]]:
    """Join chunks into the numbered context block given to the LLM.

    Stops adding chunks once max_chars is reached (small local models have
    small context windows). Returns the text AND the chunks actually included,
    so the cited sources always match what the LLM really saw.
    """
    parts: list[str] = []
    included: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        label = f"[Source {len(included) + 1}] file: {chunk.filename}"
        if chunk.page is not None:
            label += f", page {chunk.page}"
        text = chunk.document.page_content
        block = f"{label}\n{text}"
        if used + len(block) > max_chars:
            if not included:  # always keep at least the best chunk
                block = block[:max_chars]
            else:
                break
        parts.append(block)
        included.append(chunk)
        used += len(block)
    return "\n\n---\n\n".join(parts), included


def format_sources(chunks: list[RetrievedChunk], snippet_chars: int = 400) -> list[dict[str, Any]]:
    """Plain-dict sources for the UI and for the report's Sources section."""
    sources: list[dict[str, Any]] = []
    for number, chunk in enumerate(chunks, start=1):
        text = chunk.document.page_content
        snippet = text if len(text) <= snippet_chars else text[:snippet_chars].rstrip() + " …"
        sources.append(
            {
                "label": f"Source {number}",
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 4),
                "snippet": snippet,
                "text": text,
            }
        )
    return sources
