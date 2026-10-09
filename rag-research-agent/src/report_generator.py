"""Structured research report generation."""
from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings
from src.llm import safe_invoke
from src.prompts import REPORT_PROMPT


def _strip_fences(text: str) -> str:
    """Remove a ```markdown ... ``` wrapper some models add."""
    stripped = text.strip()
    match = re.match(r"^```(?:markdown|md)?\s*\n(.*?)\n```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def build_sources_section(sources: list[dict[str, Any]]) -> str:
    """Section 8 is written by code from retrieval metadata, never by the LLM,
    so the citation list cannot contain invented references."""
    if not sources:
        return "## 8. Sources\n\nNo sources were retrieved."
    lines = ["## 8. Sources", ""]
    for source in sources:
        page = f", page {source['page']}" if source.get("page") is not None else ""
        lines.append(
            f"- **[{source['label']}]** {source['filename']}{page} "
            f"(chunk `{source['chunk_id']}`, relevance {source['score']:.2f})"
        )
    return "\n".join(lines)


def assemble_report(question: str, body: str, sources: list[dict[str, Any]]) -> str:
    """Title + LLM-written sections 1-7 + code-written Sources section."""
    body = _strip_fences(body)
    body = re.split(r"(?im)^##\s*8\.", body)[0].strip()  # drop any LLM-made section 8
    body = re.sub(r"(?im)^#\s*research report\s*$", "", body).strip()
    return (
        f"# Research Report\n\n**Research question:** {question.strip()}\n\n"
        f"{body}\n\n{build_sources_section(sources)}\n"
    )


def generate_report(
    llm: BaseChatModel,
    settings: Settings,
    question: str,
    context: str,
    draft_answer: str,
    sources: list[dict[str, Any]],
) -> str:
    """Generate the 8-section report from retrieved context."""
    messages = REPORT_PROMPT.format_messages(
        question=question, context=context, draft_answer=draft_answer
    )
    body = safe_invoke(llm, messages, settings)
    return assemble_report(question, body, sources)
