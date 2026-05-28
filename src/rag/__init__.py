"""
RAG Module — Retrieval-Augmented Generation

Provides LLM-based answer generation augmented with retrieved documents.
Includes LLM provider abstraction, prompt engineering, and citation management.

Core components:
- LLMProvider: Abstract interface for LLM backends
- PromptTemplate: Prompt generation strategies
- CitationExtractor: Citation parsing and enrichment
- OutputParser: Structured response parsing
- RAGModule: Main orchestrator

Example:
    from rag.rag_module import RAGModule
    from rag.llm_provider import OllamaProvider
    
    llm = OllamaProvider(model="llama3.2:latest")
    rag = RAGModule(llm)
    response = rag.generate("How does LSI work?")
"""

__version__ = "0.1.0"

from .llm_provider import LLMProvider, OllamaProvider
from .prompt_templates import PromptTemplate, PromptTemplateFactory
from .citations import CitationExtractor, Citation
from .output_parser import OutputParser, RAGResponse

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "PromptTemplate",
    "PromptTemplateFactory",
    "CitationExtractor",
    "OutputParser",
    "RAGResponse",
    "Citation",
]
