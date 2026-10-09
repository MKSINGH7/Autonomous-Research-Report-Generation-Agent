"""Shared helpers: user-facing exceptions, logging, text cleaning, math."""
from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Sequence


# --------------------------------------------------------------------------
# Exceptions. Every RAGError message is written to be shown directly to a user.
# --------------------------------------------------------------------------
class RAGError(Exception):
    """Base class for all expected, user-presentable errors."""


class ConfigError(RAGError):
    """Invalid or missing configuration."""


class DocumentLoadError(RAGError):
    """A document could not be read."""


class EmptyDocumentError(DocumentLoadError):
    """A document contains no extractable text."""


class EmbeddingError(RAGError):
    """The embedding model failed to load or run."""


class VectorStoreError(RAGError):
    """The vector database failed."""


class EmptyQueryError(RAGError):
    """The user submitted an empty question."""


class NoDocumentsIndexedError(RAGError):
    """The vector store has no documents yet."""


class LLMError(RAGError):
    """Generic language-model failure."""


class LLMConnectionError(LLMError):
    """Cannot reach the LLM (Ollama not running, network down)."""


class LLMModelNotFoundError(LLMError):
    """The requested model does not exist / is not installed."""


class LLMAuthError(LLMError):
    """Missing or invalid API key."""


class LLMRateLimitError(LLMError):
    """Provider rate limit or quota exceeded."""


# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root logger once."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logging.getLogger(name)


def clean_text(text: str) -> str:
    """Normalise extracted text so it chunks well.

    - removes NUL characters and unifies line endings
    - re-joins words hyphenated across line breaks ("infor-\\nmation")
    - turns single line breaks (PDF layout) into spaces, keeps blank-line paragraphs
    - collapses repeated spaces and blank lines
    """
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_sha256(path: Path) -> str:
    """Content hash of a file; used to make chunk IDs stable."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """cos(theta) = (a . b) / (|a| |b|); 1.0 = same direction."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)
