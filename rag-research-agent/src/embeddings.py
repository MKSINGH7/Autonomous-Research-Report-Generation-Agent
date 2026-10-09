"""Local embedding model (Sentence-Transformers via langchain-huggingface)."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import EmbeddingError, RAGError, get_logger

logger = get_logger(__name__)


class SafeEmbeddings(Embeddings):
    """Delegates to another Embeddings object and converts any failure into an
    EmbeddingError, so the UI can tell 'embedding failed' from 'database failed'."""

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._inner.embed_documents(texts)
        except RAGError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding the document chunks failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._inner.embed_query(text)
        except RAGError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding the query failed: {exc}") from exc


def build_embeddings(model_name: str, device: str = "cpu") -> SafeEmbeddings:
    """Load a Sentence-Transformers model. The first run downloads it from
    Hugging Face (~90 MB for all-MiniLM-L6-v2); later runs use the local cache.

    normalize_embeddings=True makes every vector unit length, so cosine
    similarity equals the dot product.
    """
    try:
        inner = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load embedding model '{model_name}'. The first run needs "
            f"internet access to download it. Details: {exc}"
        ) from exc
    logger.info("Embedding model ready: %s on %s", model_name, device)
    return SafeEmbeddings(inner)


def get_embedding_dimension(embeddings: Embeddings) -> int:
    """Number of floats in one embedding (384 for all-MiniLM-L6-v2)."""
    return len(embeddings.embed_query("dimension probe"))
