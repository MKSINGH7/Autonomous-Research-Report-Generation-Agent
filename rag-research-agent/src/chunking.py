"""Split documents into overlapping chunks and attach chunk metadata."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import ConfigError


def chunk_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split page/section Documents into chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then lines, then spaces, so chunks tend to end at natural boundaries.
    Chunks never span PDF pages, which keeps page citations exact.

    Each chunk gets: chunk_index (per file), chunk_id (stable, unique),
    start_index (character offset inside its page/section).
    """
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    pieces = splitter.split_documents(documents)

    counters: dict[str, int] = {}
    chunks: list[Document] = []
    for piece in pieces:
        text = piece.page_content.strip()
        if not text:
            continue
        file_hash = str(piece.metadata.get("file_hash", "nohash"))
        index = counters.get(file_hash, 0)
        counters[file_hash] = index + 1
        metadata = dict(piece.metadata)
        metadata["chunk_index"] = index
        metadata["chunk_id"] = f"{file_hash[:12]}-{index:05d}"
        chunks.append(Document(page_content=text, metadata=metadata))
    return chunks
