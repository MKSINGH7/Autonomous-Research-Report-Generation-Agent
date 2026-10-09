"""Persistent local vector database (Chroma, cosine distance)."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.utils import RAGError, VectorStoreError, get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Thin wrapper around Chroma exposing only what this project needs.

    Chroma writes to `persist_dir` automatically, so data survives restarts and
    the existing collection is simply re-opened on startup.
    """

    def __init__(self, persist_dir: Path, collection_name: str, embeddings: Embeddings) -> None:
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._store = self._open()

    def _open(self) -> Chroma:
        try:
            return Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self._persist_dir),
                # "cosine" makes returned relevance = 1 - cosine distance.
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not open the vector database at '{self._persist_dir}': {exc}"
            ) from exc

    # ------------------------------------------------------------------ write
    def add_chunks(self, chunks: list[Document], batch_size: int = 64) -> int:
        """Embed and store chunks. Any previous chunks of the same files are
        removed first, so re-indexing a file never leaves stale chunks."""
        if not chunks:
            return 0
        try:
            for source in {str(c.metadata.get("source")) for c in chunks}:
                self.delete_source(source)
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                ids = [c.metadata["chunk_id"] for c in batch]
                self._store.add_documents(batch, ids=ids)
        except RAGError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Writing to the vector database failed: {exc}") from exc
        logger.info("Stored %d chunks", len(chunks))
        return len(chunks)

    def delete_source(self, source: str) -> int:
        """Delete every chunk that came from one file. Returns how many."""
        try:
            data = self._store.get(where={"source": source}, include=["metadatas"])
            ids = data.get("ids") or []
            if ids:
                self._store.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Deleting '{source}' failed: {exc}") from exc

    def reset(self) -> None:
        """Delete everything in this collection."""
        try:
            self._store.delete_collection()
        except Exception as exc:
            raise VectorStoreError(f"Clearing the vector database failed: {exc}") from exc
        self._store = self._open()

    # ------------------------------------------------------------------- read
    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return (chunk, relevance 0..1) pairs, best first.

        Internally: embed the query, then approximate-nearest-neighbour search.
        """
        try:
            return self._store.similarity_search_with_relevance_scores(query, k=k)
        except RAGError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Searching the vector database failed: {exc}") from exc

    def list_sources(self) -> dict[str, int]:
        """Map filename -> number of stored chunks."""
        try:
            data = self._store.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Reading the vector database failed: {exc}") from exc
        counts: dict[str, int] = {}
        for metadata in data.get("metadatas") or []:
            name = str((metadata or {}).get("source", "unknown"))
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def count(self) -> int:
        """Total number of stored chunks."""
        return sum(self.list_sources().values())
