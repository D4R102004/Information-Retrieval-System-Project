"""Session state helpers for the frontend."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from .config import (
    DEFAULT_AUTO_RELOAD,
    DEFAULT_ENABLE_WEB_SEARCH,
    DEFAULT_MAX_LOCAL_RESULTS,
    DEFAULT_MAX_WEB_RESULTS,
    DEFAULT_USE_INITIAL_CORPUS,
    DEFAULT_MIN_DOCUMENTS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_RAG_TEMPLATE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RAG_TEMPERATURE,
    DEFAULT_RAG_MAX_TOKENS,
    DEFAULT_MAX_CITES,
    DEFAULT_MAX_CONTEXT_DOC_LENGTH,
    DEFAULT_MAX_ARTICLES_PER_SPIDER,
    DEFAULT_FORCE_RECRAWL,
    DEFAULT_CLEAR_RAW
)


@dataclass
class UIState:
    """Store the user-facing UI state for a single Gradio session."""

    last_query: str = ""
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    rag_response: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(
        default_factory=lambda: {
            # Search
            "auto_reload": DEFAULT_AUTO_RELOAD,
            "enable_web_search": DEFAULT_ENABLE_WEB_SEARCH,
            "max_local_results": DEFAULT_MAX_LOCAL_RESULTS,
            "max_web_results": DEFAULT_MAX_WEB_RESULTS,
            "min_documents": DEFAULT_MIN_DOCUMENTS,

            # LLM
            "ollama_model": DEFAULT_OLLAMA_MODEL,
            "ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
            "ollama_timeout": DEFAULT_OLLAMA_TIMEOUT,
            "rag_template": DEFAULT_RAG_TEMPLATE,
            "rag_temperature": DEFAULT_RAG_TEMPERATURE,
            "rag_max_tokens": DEFAULT_RAG_MAX_TOKENS,
            "max_cites": DEFAULT_MAX_CITES,
            "max_context_doc_length": DEFAULT_MAX_CONTEXT_DOC_LENGTH,

            # Crawlers
            "max_articles_per_spider": DEFAULT_MAX_ARTICLES_PER_SPIDER,
            "force_recrawl": DEFAULT_FORCE_RECRAWL,

            # DB
            "use_initial_corpus": DEFAULT_USE_INITIAL_CORPUS,
            "clear_raw": DEFAULT_CLEAR_RAW
        }
    )
    
    def get_settings(self, particular: Literal["search", "llm", "crawler", "db", "all"]) -> dict[str, Any]:
        """
        Returns specified settings dict

        Args:
            particular (Literal["search", "llm", "crawler", "db", "all"]): particular settings to return

        Returns:
            dict[str, Any]: settings
        """

        all_settings: dict[str, Any] = {
            # Search
            "search": {
                "enable_web_search": self.settings["enable_web_search"],
                "auto_reload": self.settings["auto_reload"],
                "max_local_results": self.settings["max_local_results"],
                "max_web_results": self.settings["max_web_results"],
                "min_documents": self.settings["min_documents"],
                "query_min_length": self.settings["query_min_length"],
                "query_max_length": self.settings["query_max_length"],
            },

            # LLM
            "llm": {
                "ollama_model": self.settings["ollama_model"],
                "ollama_base_url": self.settings["ollama_base_url"],
                "ollama_timeout": self.settings["ollama_timeout"],
                "rag_temperature": self.settings["rag_temperature"],
                "rag_max_tokens": self.settings["rag_max_tokens"],
                "max_cites": self.settings["max_cites"],
                "max_rag_context_doc_length": self.settings["max_rag_context_doc_length"],
            },

            # Crawlers
            "crawler": {
                "max_articles_per_spider": self.settings["max_articles_per_spider"],
                "force_recrawl": self.settings["force_recrawl"],
            },

            # DB
            "db": {
                "use_initial_corpus": self.settings["use_initial_corpus"],
                "clear_raw": self.settings["clear_raw"]
            },
        }

        if particular == "all":
            # Flatten all categories into one dict
            merged: dict[str, Any] = {}
            for section in all_settings.values():
                merged.update(section)
            return merged

        return all_settings.get(particular, {})

def create_default_state() -> UIState:
    """Create a fresh UI state instance with default settings."""
    return UIState()


def state_to_dict(state: UIState) -> dict[str, Any]:
    """Convert a UI state object to a plain dictionary."""
    return asdict(state)
