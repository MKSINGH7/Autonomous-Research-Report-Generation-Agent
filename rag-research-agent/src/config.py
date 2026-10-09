"""Central configuration, loaded from environment variables / a .env file."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.utils import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_PROVIDERS: tuple[str, ...] = ("ollama", "gemini")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, but it is set to '{raw}'.") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, but it is set to '{raw}'.") from exc


def _resolve_path(value: str) -> Path:
    """Relative paths are resolved against the project root, not the cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """All runtime settings. Immutable; use dataclasses.replace() to override."""

    llm_provider: str
    ollama_model: str
    ollama_base_url: str
    ollama_num_ctx: int
    gemini_model: str
    gemini_api_key: str = field(repr=False)  # repr=False: never printed in logs
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    vector_db_path: Path = PROJECT_ROOT / "data" / "vectorstore"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_similarity: float = 0.15
    max_context_chars: int = 12000
    temperature: float = 0.1
    max_file_mb: int = 50

    @property
    def collection_name(self) -> str:
        """One Chroma collection per embedding model.

        Vectors from different models have different dimensions and meanings,
        so they must never share a collection.
        """
        digest = hashlib.sha1(self.embedding_model.encode("utf-8")).hexdigest()[:10]
        return f"rag_{digest}"

    def validate(self) -> None:
        """Raise ConfigError for impossible combinations."""
        if self.llm_provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER='{self.llm_provider}' is not supported. "
                f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
            )
        if self.chunk_size < 100:
            raise ConfigError("CHUNK_SIZE must be at least 100 characters.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
        if not 1 <= self.top_k <= 50:
            raise ConfigError("TOP_K must be between 1 and 50.")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ConfigError("MIN_SIMILARITY must be between 0.0 and 1.0.")
        if self.max_context_chars < 500:
            raise ConfigError("MAX_CONTEXT_CHARS must be at least 500.")


def load_settings() -> Settings:
    """Read .env (if present) and environment variables into a Settings object."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(
        llm_provider=_env_str("LLM_PROVIDER", "ollama").lower(),
        ollama_model=_env_str("OLLAMA_MODEL", "llama3.2"),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_num_ctx=_env_int("OLLAMA_NUM_CTX", 4096),
        gemini_model=_env_str("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        gemini_api_key=_env_str("GEMINI_API_KEY", _env_str("GOOGLE_API_KEY", "")),
        embedding_model=_env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_device=_env_str("EMBEDDING_DEVICE", "cpu"),
        vector_db_path=_resolve_path(_env_str("VECTOR_DB_PATH", "./data/vectorstore")),
        documents_dir=_resolve_path(_env_str("DOCUMENTS_DIR", "./data/documents")),
        top_k=_env_int("TOP_K", 5),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        min_similarity=_env_float("MIN_SIMILARITY", 0.15),
        max_context_chars=_env_int("MAX_CONTEXT_CHARS", 12000),
        temperature=_env_float("TEMPERATURE", 0.1),
        max_file_mb=_env_int("MAX_FILE_MB", 50),
    )
    settings.validate()
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    return settings
