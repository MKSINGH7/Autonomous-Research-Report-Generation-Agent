"""LLM provider registry. This is the ONLY module that knows which LLM is used.

To add a provider (e.g. OpenAI): write `_build_openai(settings)` returning a
LangChain chat model, add it to PROVIDERS and to SUPPORTED_PROVIDERS in config.py,
and add its error keywords to translate_llm_error().
"""
from __future__ import annotations

import re
from typing import Any, Callable

import requests
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings
from src.utils import (
    ConfigError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    RAGError,
    get_logger,
)

logger = get_logger(__name__)


# ------------------------------------------------------------------ builders
def _build_ollama(settings: Settings) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ConfigError("Package 'langchain-ollama' is not installed.") from exc
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        # Ollama's default context window is small and silently truncates long
        # prompts, so we set it explicitly.
        num_ctx=settings.ollama_num_ctx,
    )


def _build_gemini(settings: Settings) -> BaseChatModel:
    if not settings.gemini_api_key:
        raise LLMAuthError(
            "GEMINI_API_KEY is not set. Add it to your .env file (or paste it in "
            "the sidebar), or switch LLM_PROVIDER to 'ollama'."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ConfigError("Package 'langchain-google-genai' is not installed.") from exc
    # Temperature is intentionally left at the library default: Google advises
    # keeping the default for Gemini 3 models.
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        max_retries=2,
    )


PROVIDERS: dict[str, Callable[[Settings], BaseChatModel]] = {
    "ollama": _build_ollama,
    "gemini": _build_gemini,
}


def get_llm(settings: Settings) -> BaseChatModel:
    """Return the chat model selected by settings.llm_provider."""
    builder = PROVIDERS.get(settings.llm_provider)
    if builder is None:
        raise ConfigError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            f"Available: {', '.join(PROVIDERS)}."
        )
    return builder(settings)


def active_model_name(settings: Settings) -> str:
    return settings.ollama_model if settings.llm_provider == "ollama" else settings.gemini_model


# -------------------------------------------------------------------- Ollama
def list_ollama_models(base_url: str, timeout: float = 3.0) -> list[str]:
    """Names of models installed in the local Ollama server."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except requests.exceptions.RequestException as exc:
        raise LLMConnectionError(
            f"Cannot reach Ollama at {base_url}. Start the Ollama app (or run "
            f"'ollama serve') and try again. Details: {exc}"
        ) from exc
    except (ValueError, KeyError) as exc:
        raise LLMError(f"Unexpected reply from Ollama at {base_url}: {exc}") from exc


def _model_installed(name: str, installed: list[str]) -> bool:
    wanted = name if ":" in name else f"{name}:latest"
    return wanted in installed or name in installed


# ------------------------------------------------------------ error handling
def translate_llm_error(exc: Exception, settings: Settings) -> LLMError:
    """Convert provider-specific exceptions into friendly LLMError subclasses."""
    if isinstance(exc, LLMError):
        return exc
    text = str(exc)
    low = text.lower()
    model = active_model_name(settings)

    def has(*words: str) -> bool:
        return any(w in low for w in words)

    def has_code(*codes: str) -> bool:
        return any(re.search(rf"\b{c}\b", low) for c in codes)

    connection_hint = has("connect", "connection", "refused", "timed out", "timeout", "dns", "unreachable")

    if settings.llm_provider == "ollama":
        if isinstance(exc, ConnectionError) or connection_hint:
            return LLMConnectionError(
                f"Cannot reach Ollama at {settings.ollama_base_url}. Start the "
                "Ollama app (or run 'ollama serve')."
            )
        if (has("not found") and has("model")) or has("try pulling"):
            return LLMModelNotFoundError(
                f"Ollama model '{model}' is not installed. Run:  ollama pull {model}"
            )
        if has("memory"):
            return LLMError(
                f"Ollama could not load '{model}': not enough RAM. Use a smaller "
                f"model (e.g. llama3.2:1b) or close other programs. Details: {text[:200]}"
            )
    else:
        if has("api key not valid", "api_key_invalid", "invalid api key", "permission_denied", "unauthenticated") or has_code("401", "403"):
            return LLMAuthError(
                "The Gemini API rejected your API key. Check GEMINI_API_KEY in .env "
                "(no quotes/spaces) and that the key is enabled in Google AI Studio."
            )
        if has("resource_exhausted", "quota", "rate limit", "too many requests") or has_code("429"):
            return LLMRateLimitError(
                "Gemini rate limit or quota reached. Wait a minute and retry, or "
                "switch to local Ollama mode."
            )
        if (has_code("404") or has("is not found", "not supported for generatecontent")) and has("model"):
            return LLMModelNotFoundError(
                f"Gemini model '{model}' was not found. Model names change; set "
                "GEMINI_MODEL to a current name from Google AI Studio."
            )
        if connection_hint:
            return LLMConnectionError(
                "Could not connect to the Gemini API. Check your internet connection, "
                "VPN or firewall."
            )
    return LLMError(f"The language model call failed: {text[:300]}")


def message_to_text(message: Any) -> str:
    """Extract plain text from a LangChain message (content may be str or blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def safe_invoke(llm: BaseChatModel, messages: Any, settings: Settings) -> str:
    """Call the LLM and return text; all failures become friendly LLMErrors."""
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        raise translate_llm_error(exc, settings) from exc
    text = message_to_text(response).strip()
    if not text:
        raise LLMError("The model returned an empty response. Try again or use another model.")
    return text


def check_llm_connection(settings: Settings) -> tuple[bool, str]:
    """Small end-to-end test used by the sidebar 'Test LLM connection' button."""
    try:
        if settings.llm_provider == "ollama":
            installed = list_ollama_models(settings.ollama_base_url)
            if not _model_installed(settings.ollama_model, installed):
                have = ", ".join(installed) or "none"
                return False, (
                    f"Ollama is running but model '{settings.ollama_model}' is not "
                    f"installed (installed: {have}). Run: ollama pull {settings.ollama_model}"
                )
        llm = get_llm(settings)
        reply = safe_invoke(llm, "Reply with the single word OK.", settings)
        return True, (
            f"Connected: {settings.llm_provider} / {active_model_name(settings)}. "
            f"Test reply: {reply[:60]}"
        )
    except RAGError as exc:
        return False, str(exc)
