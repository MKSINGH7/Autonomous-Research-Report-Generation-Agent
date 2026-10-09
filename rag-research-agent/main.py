"""Streamlit UI for the Autonomous Research & Report Generation Agent."""
import dataclasses
import traceback
from pathlib import Path

import streamlit as st
from src.config import SUPPORTED_PROVIDERS, Settings, load_settings
from src.embeddings import build_embeddings
from src.graph import build_ingestion_graph, build_research_graph, run_ingestion, run_research
from src.llm import (
    active_model_name,
    check_llm_connection,
    get_llm,
    list_ollama_models,
)
from src.loaders import SUPPORTED_EXTENSIONS
from src.retriever import Retriever
from src.utils import ConfigError, RAGError, get_logger
from src.vectorstore import VectorStoreManager

logger = get_logger("app")


# ------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)…")
def cached_embeddings(model_name: str, device: str):
    return build_embeddings(model_name, device)


@st.cache_resource(show_spinner="Opening vector database…")
def cached_store(persist_dir: str, collection: str, model_name: str, device: str):
    return VectorStoreManager(Path(persist_dir), collection, cached_embeddings(model_name, device))


@st.cache_data(ttl=15, show_spinner=False)
def cached_ollama_models(base_url: str) -> list[str]:
    try:
        return list_ollama_models(base_url)
    except RAGError:
        return []


def show_unexpected(exc: Exception) -> None:
    logger.exception("Unexpected error")
    st.error("Something unexpected went wrong. Details are below if you need to report it.")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(exc)))


# --------------------------------------------------------------------- sidebar
def sidebar_settings(base: Settings) -> Settings:
    st.sidebar.header("Configuration")
    provider = st.sidebar.selectbox(
        "LLM provider",
        SUPPORTED_PROVIDERS,
        index=SUPPORTED_PROVIDERS.index(base.llm_provider),
        help="ollama = local & free. gemini = online API (needs an API key).",
    )
    updates: dict = {"llm_provider": provider}

    if provider == "ollama":
        installed = cached_ollama_models(base.ollama_base_url)
        if installed:
            options = list(installed)
            default = base.ollama_model if ":" in base.ollama_model else f"{base.ollama_model}:latest"
            if default not in options:
                options.insert(0, base.ollama_model)
            updates["ollama_model"] = st.sidebar.selectbox(
                "Ollama model", options, index=options.index(default) if default in options else 0
            )
        else:
            st.sidebar.warning("Ollama is not reachable (or has no models). Start Ollama and pull a model.")
            updates["ollama_model"] = st.sidebar.text_input("Ollama model", base.ollama_model)
    else:
        updates["gemini_model"] = st.sidebar.text_input("Gemini model", base.gemini_model)
        typed_key = st.sidebar.text_input(
            "Gemini API key (optional override)",
            type="password",
            help="Used for this browser session only and never saved. Prefer the .env file.",
        )
        if typed_key.strip():
            updates["gemini_api_key"] = typed_key.strip()
        elif base.gemini_api_key:
            st.sidebar.caption("API key loaded from .env / environment")
        else:
            st.sidebar.warning("No API key found. Set GEMINI_API_KEY in .env or paste it above.")

    st.sidebar.subheader("Chunking (applies to newly indexed files)")
    chunk_size = int(
        st.sidebar.number_input(
            "Chunk size (characters)", 100, 8000, min(max(base.chunk_size, 100), 8000), step=100
        )
    )
    chunk_overlap = int(
        st.sidebar.number_input(
            "Chunk overlap (characters)", 0, chunk_size - 1, min(base.chunk_overlap, chunk_size - 1), step=50
        )
    )
    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.slider("Top-K chunks", 1, 20, min(max(base.top_k, 1), 20))
    min_similarity = st.sidebar.slider(
        "Minimum similarity", 0.0, 1.0, float(base.min_similarity), 0.05,
        help="Chunks scoring below this are ignored.",
    )
    updates.update(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k, min_similarity=min_similarity
    )

    settings = dataclasses.replace(base, **updates)
    try:
        settings.validate()
    except ConfigError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    if st.sidebar.button("Test LLM connection"):
        with st.sidebar:
            with st.spinner("Contacting the model…"):
                ok, message = check_llm_connection(settings)
        (st.sidebar.success if ok else st.sidebar.error)(message)
    return settings


# ------------------------------------------------------------------- ingestion
def save_uploads(uploads, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for upload in uploads:
        name = Path(upload.name).name  # strip any path components
        target = target_dir / name
        target.write_bytes(upload.getbuffer())
        saved.append(target)
    return saved


def index_files(settings: Settings, store: VectorStoreManager, paths: list[Path]) -> None:
    try:
        with st.spinner("Loading → cleaning → chunking → embedding → storing…"):
            graph = build_ingestion_graph(settings, store)
            result = run_ingestion(graph, paths)
    except RAGError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - last-resort UI guard
        show_unexpected(exc)
    else:
        st.success(
            f"Indexed {result.get('indexed_chunks', 0)} chunks from "
            f"{len(result.get('documents', []))} page(s)/section(s)."
        )
        for warning in result.get("warnings", []):
            st.warning(warning)


def render_ingestion(settings: Settings, store: VectorStoreManager) -> None:
    st.header("1️. Upload and index documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT or DOCX files",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    col1, col2 = st.columns(2)
    if col1.button("Process / Index uploaded documents", type="primary"):
        if not uploads:
            st.warning("Please upload at least one document first.")
        else:
            index_files(settings, store, save_uploads(uploads, settings.documents_dir))
    if col2.button(f"Index all files in {settings.documents_dir.name}/ folder"):
        files = [
            p for p in sorted(settings.documents_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            st.warning(f"No supported files found in {settings.documents_dir}.")
        else:
            index_files(settings, store, files)

    try:
        sources = store.list_sources()
    except RAGError as exc:
        st.error(str(exc))
        return
    with st.expander(f"Indexed documents ({len(sources)} files, {sum(sources.values())} chunks)"):
        if not sources:
            st.info("Nothing indexed yet.")
        else:
            st.table([{"Document": n, "Chunks": c} for n, c in sources.items()])
            to_delete = st.selectbox("Remove a document", ["—"] + list(sources))
            if st.button("Remove selected") and to_delete != "—":
                try:
                    store.delete_source(to_delete)
                    st.success(f"Removed {to_delete}. Reload the page to refresh the list.")
                except RAGError as exc:
                    st.error(str(exc))
            confirm = st.checkbox("I want to delete the whole vector database")
            if st.button("Clear everything") and confirm:
                try:
                    store.reset()
                    st.success("Vector database cleared. Reload the page.")
                except RAGError as exc:
                    st.error(str(exc))


# -------------------------------------------------------------------- research
def run_question(settings: Settings, store: VectorStoreManager, question: str, want_report: bool) -> None:
    if not question.strip():
        st.warning("Please type a research question first.")
        return
    try:
        llm = get_llm(settings)
        retriever = Retriever(store, settings.top_k, settings.min_similarity)
        graph = build_research_graph(retriever, llm, settings)
        with st.spinner("Retrieving context and generating (local models can take a minute)…"):
            result = run_research(graph, question, want_report)
        st.session_state["result"] = {
            **result,
            "retrieved_documents": None,  # not needed by the UI
            "model": f"{settings.llm_provider} / {active_model_name(settings)}",
        }
    except RAGError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        show_unexpected(exc)


def render_result() -> None:
    result = st.session_state.get("result")
    if not result:
        return
    st.caption(f"Generated with {result['model']}")
    if result.get("insufficient_context"):
        st.warning("No sufficiently relevant passages were found, so the LLM was not called.")
    answer_tab, report_tab, sources_tab = st.tabs(["💬 Answer", "📄 Report", "🔍 Sources"])
    with answer_tab:
        st.markdown(result.get("answer", ""))
    with report_tab:
        if result.get("report"):
            st.markdown(result["report"])
            st.download_button(
                "Download report (.md)", result["report"], "research_report.md", "text/markdown"
            )
        else:
            st.info("No report was generated (tick the report option, or no relevant context).")
    with sources_tab:
        sources = result.get("sources") or []
        if not sources:
            st.info("No sources retrieved.")
        for source in sources:
            page = f", page {source['page']}" if source.get("page") is not None else ""
            title = f"{source['label']} — {source['filename']}{page} — relevance {source['score']:.2f}"
            with st.expander(title):
                st.progress(min(max(float(source["score"]), 0.0), 1.0))
                st.caption(f"Chunk ID: {source['chunk_id']}")
                st.text(source["text"])


def render_research(settings: Settings, store: VectorStoreManager) -> None:
    st.header("2️. Ask a research question")
    question = st.text_area(
        "Research question", height=100,
        placeholder="e.g. What are the main limitations of the proposed method?",
    )
    want_report = st.checkbox("Also generate a structured research report", value=True)
    if st.button("Research", type="primary"):
        run_question(settings, store, question, want_report)
    render_result()


# ------------------------------------------------------------------------ main
def main() -> None:
    st.set_page_config(page_title="Autonomous Research Agent", layout="wide")
    st.title("Autonomous Research & Report Generation Agent")

    try:
        base = load_settings()
    except ConfigError as exc:
        st.error(f"Configuration problem: {exc}")
        st.stop()

    settings = sidebar_settings(base)
    try:
        store = cached_store(
            str(settings.vector_db_path),
            settings.collection_name,
            settings.embedding_model,
            settings.embedding_device,
        )
    except RAGError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        show_unexpected(exc)
        st.stop()

    render_ingestion(settings, store)
    st.divider()
    render_research(settings, store)


main()
