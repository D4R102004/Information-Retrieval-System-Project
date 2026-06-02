"""Session state helpers for the frontend."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from src.main_config import MainConfig

# from .config import (
#     DEFAULT_AUTO_RELOAD,
#     DEFAULT_ENABLE_WEB_SEARCH,
#     DEFAULT_MAX_LOCAL_RESULTS,
#     DEFAULT_MAX_WEB_RESULTS,
#     DEFAULT_USE_INITIAL_CORPUS,
#     DEFAULT_MIN_DOCUMENTS,
#     DEFAULT_QUERY_MIN_LENGTH,
#     DEFAULT_QUERY_MAX_LENGTH,
#     DEFAULT_OLLAMA_MODEL,
#     DEFAULT_RAG_TEMPLATE,
#     DEFAULT_OLLAMA_BASE_URL,
#     DEFAULT_RAG_TEMPERATURE,
#     DEFAULT_RAG_MAX_TOKENS,
#     DEFAULT_MAX_CITES,
#     DEFAULT_MAX_CONTEXT_DOC_LENGTH,
#     DEFAULT_MAX_ARTICLES_PER_SPIDER,
#     DEFAULT_FORCE_RECRAWL,
# )


@dataclass
class UIState:
    """Store the user-facing UI state for a single Gradio session."""

    _backend_config = MainConfig()
    last_query: str = ""
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    rag_response: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(
        default_factory=lambda: {
            # Search
            "enable_web_search": MainConfig()["enable_web_search"],
            "auto_reload": MainConfig()["auto_reload"],
            "max_local_results": MainConfig()["max_local_results"],
            "max_web_results": MainConfig()["max_web_results"],
            "min_documents": MainConfig()["min_documents"],

            # LLM
            "ollama_model": MainConfig()["ollama_model"],
            "ollama_base_url": MainConfig()["ollama_base_url"],
            "rag_template": MainConfig()["rag_template"],
            "rag_temperature": MainConfig()["rag_temperature"],
            "rag_max_tokens": MainConfig()["rag_max_tokens"],
            "max_cites": MainConfig()["max_cites"],
            "max_context_doc_length": MainConfig()["max_context_doc_length"],

            # Crawlers
            "max_articles_per_spider": MainConfig()["max_articles_per_spider"],
            "force_recrawl": MainConfig()["force_recrawl"],

            # DB
            "use_initial_corpus": MainConfig()["use_initial_corpus"],
            "clear_raw": MainConfig()["clear_raw"]
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
                "clear_raw": MainConfig()["clear_raw"]
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

def sync_backend(state: UIState) -> bool:
    """Attempts to sync settings with backend configuration

    Args:
        state (UIState): current state

    Returns:
        bool: Syncronization performed correctly
    """
    backend_config = MainConfig()
    try:
        for key, val in state.settings.items():
            backend_config[key] = val
        return True
    except:
        return False
