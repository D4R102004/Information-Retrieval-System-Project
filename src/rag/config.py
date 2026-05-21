"""
RAG Module Configuration

Centralizes all RAG configuration parameters.
Supports environment variables for production deployment.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RAGConfig(BaseSettings):
    """RAG module configuration from environment variables."""

    # LLM Configuration
    ollama_model: str = "llama3.2:latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300

    # RAG Configuration
    rag_template: str = "domain_specific"  # basic, domain_specific, chain_of_thought
    rag_retrieval_top_k: int = 5
    rag_temperature: float = 0.7
    rag_max_tokens: int = 1024
    rag_citation_threshold: float = 0.0

    # Logging
    rag_log_level: str = "INFO"

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def validate_template(self) -> None:
        """Validate that template is valid."""
        valid_templates = ["basic", "domain_specific", "chain_of_thought"]
        if self.rag_template not in valid_templates:
            raise ValueError(
                f"Invalid template: {self.rag_template}. "
                f"Must be one of: {valid_templates}"
            )

    def __init__(self, **data):
        """Initialize config and validate."""
        super().__init__(**data)
        self.validate_template()
        logger.info(f"RAG Config loaded: template={self.rag_template}, ollama_model={self.ollama_model}")


# Global config instance
config = RAGConfig()
