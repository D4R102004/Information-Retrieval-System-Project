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
import re

from .config import config as rag_config

logger = logging.getLogger(__name__)
EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    """Remove common emoji symbols from text."""
    return EMOJI_PATTERN.sub("", text)


class PromptTemplate(ABC):
    """Abstract base for prompt generation strategies."""

    json_response = """IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "answer": "Your detailed answer here with inline citations [doc_id]",
  "citations": ["doc_id1", "doc_id2", ...]
}}

Ensure the JSON is properly formatted with escaped quotes and no trailing commas.

JSON Response:"""

    @abstractmethod
    def apply(self, query: str, documents: List[Dict], require_json: bool = False) -> List[str]:
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
        self, documents: List[Dict], max_chars: int = rag_config.max_doc_content_length * 5
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
            if doc_id == "unknown":                     # Legacy name in case of normalization
                doc_id = doc.get("doc_id", "unknown")   # failure when receiving documents
            title = _strip_emojis(str(doc.get("title", "Untitled")))
            content = _strip_emojis(str(doc.get("content", "")))[:rag_config.max_doc_content_length]

            doc_text = f"ID: [{doc_id}]\nTITLE: {title}\nCONTENT:\n{content}\n"

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

    def apply(self, query: str, documents: List[Dict], require_json: bool = False) -> List[str]:
        """Generate basic prompt."""
        context = self._format_context(documents, max_chars=2000)

        return [f"""Context:
{context}

Question: {query}

{self.json_response if require_json else "Answer:"}"""]


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
            "Your role is to provide simple, accurate, informative answers based on the provided documents. "
        )
        logger.debug("Initialized DomainSpecificTemplate")

    def apply(self, query: str, documents: List[Dict], require_json: bool = False) -> List[str]:
        """Generate domain-specific prompt."""
        context = self._format_context(documents, max_chars=4000)

        return [f"""{self.system_prompt}

## Available Documents:
{context}

## User Question:
{query}

## Instructions:
- Provide a comprehensive answer based on the documents
- Always cite sources plainly as [doc_id] when referencing specific information (e.g. "Python is a programming language [doc_1] used in...")
- If information is not in the documents, state that clearly
- Be precise, and avoid speculation and redundancy
- Only refer to a document when citing it, avoid referencing it or its title directly in your answer
- Only use [] when citing, do not use in any other case

## {self.json_response if require_json else "Answer:"}"""]


class ChainOfThoughtTemplate(PromptTemplate):
    """
    Chain-of-Thought reasoning template.

    Trade-off: Best quality, higher token usage, explicit reasoning.

    Encourages step-by-step reasoning, improving answer quality and
    consistency. Slightly higher token consumption but better results.
    """
    def __init__(self):
        """Initialize domain-specific template."""
        self.system_prompt = (
            "You are a technical assistant specialized in software and technology. "
            "Your role is to provide simple, accurate, informative answers based on the provided documents. "
        )
        logger.debug("Initialized ChainOfThoughtTemplate")

    def apply(self, query: str, documents: List[Dict], require_json: bool = False) -> List[str]:
        """Generate chain-of-thought prompt."""
        context = self._format_context(documents, max_chars=5000)

        return [f"""{self.system_prompt}

## Available Documents:
{context}

## User Question:
{query}

## Reasoning Process:

Think through this step-by-step:

1. **Document Relevance Analysis:** Which documents contain information relevant to this question?

2. **Information Synthesis:** How should information from multiple documents be combined?

3. **Source Attribution:** Which specific documents support each claim?

4. **Answer Construction:** Generate a comprehensive answer with proper citations based on the analysis.
{"\n5. **Citation Extraction:** Extract document IDs that support your answer\n" if require_json else ""}
## Answer Format:
- Provide your answer with inline citations [doc_id] (e.g. "Python is a programming language [doc_1] used in...")
- Only use [] when citing, do not use in any other case
- Explain the reasoning when synthesizing multiple sources
- Be precise and non-redundant, and cite only when information comes from documents
- Only refer to a document when citing it, avoid referencing it or its title directly in your answer

## {self.json_response if require_json else "Answer:"}"""]


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
