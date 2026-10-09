"""LangGraph workflows: one for ingestion, one for research.

Why LangGraph instead of one big function?
  * Explicit shared STATE: every step reads/writes named fields.
  * Each step is an isolated, testable NODE.
  * CONDITIONAL EDGES: "no relevant chunks -> skip the LLM" and
    "report requested? -> generate it" are routing decisions in the graph,
    not nested if-statements.
  * The flow can be drawn (graph.get_graph().draw_mermaid()), extended
    (rerank, web search, human approval) and later given checkpointing/streaming.

Note: a node name may not equal a state key, which is why the report flag is
called `want_report` while the node is called `generate_report`.
"""
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.chunking import chunk_documents
from src.config import Settings
from src.llm import safe_invoke
from src.loaders import load_documents
from src.prompts import NO_CONTEXT_MESSAGE, RAG_PROMPT
from src.report_generator import generate_report
from src.retriever import RetrievedChunk, Retriever, build_context, format_sources
from src.utils import DocumentLoadError, EmptyDocumentError, clean_text
from src.vectorstore import VectorStoreManager


# =========================================================== INGESTION GRAPH
class IngestionState(TypedDict, total=False):
    file_paths: list[str]
    documents: list[Document]
    chunks: list[Document]
    warnings: list[str]
    indexed_chunks: int


def build_ingestion_graph(settings: Settings, store: VectorStoreManager):
    """START -> load_documents -> clean_text -> chunk_documents -> index_chunks -> END"""

    def load_node(state: IngestionState) -> dict[str, Any]:
        paths = [Path(p) for p in state.get("file_paths", [])]
        if not paths:
            raise DocumentLoadError("No files were provided.")
        documents, warnings = load_documents(paths, settings.max_file_mb)
        if not documents:
            detail = " ".join(warnings) or "No readable text was found."
            raise DocumentLoadError(f"None of the files could be processed. {detail}")
        return {"documents": documents, "warnings": warnings}

    def clean_node(state: IngestionState) -> dict[str, Any]:
        cleaned = []
        for doc in state["documents"]:
            text = clean_text(doc.page_content)
            if text:
                cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
        if not cleaned:
            raise EmptyDocumentError("The documents contain no usable text after cleaning.")
        return {"documents": cleaned}

    def chunk_node(state: IngestionState) -> dict[str, Any]:
        chunks = chunk_documents(state["documents"], settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise EmptyDocumentError("No chunks could be created from the documents.")
        return {"chunks": chunks}

    def index_node(state: IngestionState) -> dict[str, Any]:
        return {"indexed_chunks": store.add_chunks(state["chunks"])}

    builder = StateGraph(IngestionState)
    builder.add_node("load_documents", load_node)
    builder.add_node("clean_text", clean_node)
    builder.add_node("chunk_documents", chunk_node)
    builder.add_node("index_chunks", index_node)
    builder.add_edge(START, "load_documents")
    builder.add_edge("load_documents", "clean_text")
    builder.add_edge("clean_text", "chunk_documents")
    builder.add_edge("chunk_documents", "index_chunks")
    builder.add_edge("index_chunks", END)
    return builder.compile()


def run_ingestion(graph, file_paths: list[Path]) -> IngestionState:
    return graph.invoke({"file_paths": [str(p) for p in file_paths]})


# ============================================================= RESEARCH GRAPH
class ResearchState(TypedDict, total=False):
    user_query: str
    want_report: bool
    retrieved_documents: list[RetrievedChunk]
    context: str
    sources: list[dict[str, Any]]
    answer: str
    report: str
    insufficient_context: bool


def build_research_graph(retriever: Retriever, llm: BaseChatModel, settings: Settings):
    """
    START -> retrieve_context --(chunks found)--> generate_answer
                  |                                   |-(want_report)-> generate_report -> END
                  |                                   '-(otherwise)---> END
                  '--(nothing relevant)--> no_context -> END
    """

    def retrieve_node(state: ResearchState) -> dict[str, Any]:
        results = retriever.retrieve(state["user_query"])
        context, used = build_context(results, settings.max_context_chars)
        return {
            "retrieved_documents": used,
            "context": context,
            "sources": format_sources(used),
        }

    def route_after_retrieval(state: ResearchState) -> str:
        return "generate_answer" if state.get("retrieved_documents") else "no_context"

    def no_context_node(state: ResearchState) -> dict[str, Any]:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "report": "",
            "sources": [],
            "insufficient_context": True,
        }

    def answer_node(state: ResearchState) -> dict[str, Any]:
        messages = RAG_PROMPT.format_messages(
            context=state["context"], question=state["user_query"]
        )
        return {"answer": safe_invoke(llm, messages, settings), "insufficient_context": False}

    def route_after_answer(state: ResearchState) -> str:
        return "generate_report" if state.get("want_report") else "finish"

    def report_node(state: ResearchState) -> dict[str, Any]:
        report = generate_report(
            llm,
            settings,
            state["user_query"],
            state["context"],
            state["answer"],
            state.get("sources", []),
        )
        return {"report": report}

    builder = StateGraph(ResearchState)
    builder.add_node("retrieve_context", retrieve_node)
    builder.add_node("no_context", no_context_node)
    builder.add_node("generate_answer", answer_node)
    builder.add_node("generate_report", report_node)

    builder.add_edge(START, "retrieve_context")
    builder.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {"generate_answer": "generate_answer", "no_context": "no_context"},
    )
    builder.add_edge("no_context", END)
    builder.add_conditional_edges(
        "generate_answer",
        route_after_answer,
        {"generate_report": "generate_report", "finish": END},
    )
    builder.add_edge("generate_report", END)
    return builder.compile()


def run_research(graph, question: str, want_report: bool = True) -> ResearchState:
    return graph.invoke({"user_query": question, "want_report": want_report})
