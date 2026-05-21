"""
Prompt Engineering Module

Implements three prompt template strategies with increasing sophistication:
1. Basic: Minimal structure for testing
2. Domain-Specific: Role-play for technical domain (development choice)
3. Chain-of-Thought: Step-by-step reasoning (production choice)

Templates are interchangeable at runtime using Factory pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
MAX_CONTENT_SIZE = 500

class PromptTemplate(ABC):
    """Abstract base for prompt generation strategies."""

    @abstractmethod
    def apply(self, query: str, documents: List[Dict]) -> str:
        """
        Generate a prompt from query and retrieved documents.

        Args:
            query: User question
            documents: Retrieved documents with metadata
                Expected keys: id, title, content, source, url

        Returns:
            Formatted prompt ready for LLM
        """
        pass

    def _format_context(
        self, documents: List[Dict], max_chars: int = 3000
    ) -> str:
        """Format documents into context string.

        Args:
            documents: Retrieved documents
            max_chars: Maximum context size in characters

        Returns:
            Formatted context string
        """
        context = []
        total_chars = 0

        for doc in documents:
            # Build document entry
            doc_id = doc.get("id", "unknown")
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")[:MAX_CONTENT_SIZE]  # Truncate

            doc_text = f"[{doc_id}] {title}\n{content}\n"

            # Check if adding this doc would exceed limit
            if total_chars + len(doc_text) > max_chars:
                break

            context.append(doc_text)
            total_chars += len(doc_text)

        return "\n".join(context)


class BasicTemplate(PromptTemplate):
    """
    Simple template for testing.

    Trade-off: Low quality responses, minimal token usage.
    Useful for rapid prototyping and testing.
    """

    def apply(self, query: str, documents: List[Dict]) -> str:
        """Generate basic prompt."""
        context = self._format_context(documents, max_chars=2000)

        return f"""Context:
{context}

Question: {query}

Answer:"""


class DomainSpecificTemplate(PromptTemplate):
    """
    Technical assistant role-play template.

    Trade-off: Good quality, moderate token usage, domain-aligned.

    Suitable for technology and software domain. Includes role definition
    and clear instructions for citation and accuracy.
    """

    def __init__(self):
        """Initialize domain-specific template."""
        self.system_prompt = (
            "You are a technical assistant specialized in software and technology. "
            "Your role is to provide accurate, informative answers based on the provided documents. "
        )
        logger.debug("Initialized DomainSpecificTemplate")

    def apply(self, query: str, documents: List[Dict]) -> str:
        """Generate domain-specific prompt."""
        context = self._format_context(documents, max_chars=4000)

        return f"""{self.system_prompt}

## Retrieved Documents:
{context}

## User Question:
{query}

## Instructions:
- Provide a comprehensive answer based on the documents
- Always cite sources as [doc_id] when referencing specific information
- If information is not in the documents, state that clearly
- Be precise and avoid speculation

## Answer:"""


class ChainOfThoughtTemplate(PromptTemplate):
    """
    Chain-of-Thought reasoning template.

    Trade-off: Best quality, higher token usage, explicit reasoning.

    Encourages step-by-step reasoning, improving answer quality and
    consistency. Slightly higher token consumption but better results.
    """

    def apply(self, query: str, documents: List[Dict]) -> str:
        """Generate chain-of-thought prompt."""
        context = self._format_context(documents, max_chars=5000)

        return f"""You are a technical assistant specializing in software and technology.

## Available Documents:
{context}

## User Question:
{query}

## Reasoning Process:

Think through this step-by-step:

1. **Document Relevance Analysis:** Which documents contain information relevant to this question?

2. **Information Synthesis:** How should information from multiple documents be combined?

3. **Source Attribution:** Which specific documents support each claim?

4. **Answer Construction:** Generate a comprehensive answer with proper citations.

## Answer Format:
- Provide your answer with inline citations [doc_id]
- Explain the reasoning when synthesizing multiple sources
- Be precise and cite only when information comes from documents

## Answer:"""


class PromptTemplateFactory:
    """Factory for creating and managing prompt templates."""

    _templates = {
        "basic": BasicTemplate,
        "domain_specific": DomainSpecificTemplate,
        "chain_of_thought": ChainOfThoughtTemplate,
    }

    @classmethod
    def create(cls, template_type: str) -> PromptTemplate:
        """
        Create a prompt template by name.

        Args:
            template_type: Template name
                ("basic", "domain_specific", "chain_of_thought")

        Returns:
            PromptTemplate instance

        Raises:
            ValueError: If template type is unknown
        """
        if template_type not in cls._templates:
            available = ", ".join(cls._templates.keys())
            raise ValueError(
                f"Unknown template: '{template_type}'. "
                f"Available: {available}"
            )

        template_class = cls._templates[template_type]
        logger.debug(f"Creating template: {template_type}")
        return template_class()

    @classmethod
    def available_templates(cls) -> List[str]:
        """List available template types.

        Returns:
            List of template names
        """
        return list(cls._templates.keys())

    @classmethod
    def register(cls, name: str, template_class: type) -> None:
        """
        Register a custom template.

        Args:
            name: Template identifier
            template_class: PromptTemplate subclass
        """
        if not issubclass(template_class, PromptTemplate):
            raise TypeError(
                f"Template class must inherit from PromptTemplate, got {template_class}"
            )
        cls._templates[name] = template_class
        logger.info(f"Registered custom template: {name}")
