# RAG Module Implementation Plan

---

## Executive Summary

This document outlines the implementation strategy for the Retrieval-Augmented Generation (RAG) module, a mandatory component in the project. The plan integrates with existing system architecture while maintaining modularity, extensibility, and reproducibility.

**Key Design Principles:**
- Minimal coupling to retrieval pipeline
- LLM provider abstraction for future migrations
- Progressive complexity (start simple, end sophisticated)
- Full compliance with course requirements

---

## 1. Architecture Overview

### 1.1 System Integration

The RAG module operates as a separate orchestration layer built on top of the existing SRIPipeline:

```
┌─────────────────┐
│ User Query      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    RAG Module (NEW)                 │
│  ┌───────────────────────────────┐  │
│  │ 1. Query Processing           │  │
│  │ 2. Document Retrieval         │  │
│  │ 3. Prompt Engineering         │  │
│  │ 4. LLM Generation             │  │
│  │ 5. Citation Extraction        │  │
│  └───────────────────────────────┘  │
└────────┬────────────────────────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
    ┌────────────┐       ┌─────────────┐
    │ Retriever  │       │ LLM Provider│
    │ (Pipeline) │       │ (Ollama)    │
    └────────────┘       └─────────────┘
         │
         ├─► LSI Model
         ├─► Vector Store
         └─► Ranking Engine
         
    ▼
┌──────────────────────────────────┐
│ Structured Output (Answer + Citations)
└──────────────────────────────────┘
```

### 1.2 Module Responsibilities

| Component | Responsibility | Location |
|-----------|-----------------|----------|
| **RAGModule** | Orchestration, flow control, response assembly | `src/rag/rag_module.py` |
| **LLMProvider** | LLM abstraction, vendor-agnostic interface | `src/rag/llm_provider.py` |
| **LLMOllama** | Ollama-specific implementation | `src/rag/llm_provider.py` |
| **PromptTemplate** | Prompt generation strategies | `src/rag/prompt_templates.py` |
| **CitationExtractor** | Citation parsing from LLM output | `src/rag/citations.py` |
| **OutputParser** | JSON validation and repair | `src/rag/output_parser.py` |

---

## 2. LLM Provider Abstraction

### 2.1 Design Pattern: Strategy Pattern

The LLM provider follows the Strategy pattern to enable seamless migration between LLM vendors without modifying RAG logic.

#### 2.1.1 Base Interface

```python
# src/rag/llm_provider.py

from abc import ABC, abstractmethod
from typing import Optional

class LLMProvider(ABC):
    """Abstract base class for Language Model providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: Input prompt text
            temperature: Creativity parameter (0.0-1.0)
            max_tokens: Maximum output length
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is accessible."""
        pass
    
    def get_metadata(self) -> dict:
        """Return provider metadata (model name, version, etc.)"""
        return {}
```

#### 2.1.2 Ollama Implementation

```python
# src/rag/llm_provider.py (continuation)

import requests
import os
from typing import Optional

class OllamaProvider(LLMProvider):
    """Interface to Ollama local LLM service."""
    
    def __init__(
        self,
        model: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        verify_ssl: bool = False,
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model identifier (default: llama3.2:latest)
            base_url: Ollama service endpoint
            timeout: Request timeout in seconds
            verify_ssl: SSL verification for requests
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Verify that the specified model is available in Ollama."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            available_models = [m["name"] for m in response.json().get("models", [])]
            
            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' not found. Available: {available_models}"
                )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? (ollama serve)"
            )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
    ) -> str:
        """Generate text using Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2,
                verify=self.verify_ssl,
            )
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    
    def get_metadata(self) -> dict:
        """Return Ollama metadata."""
        return {
            "provider": "Ollama",
            "model": self.model,
            "base_url": self.base_url,
            "type": "local",
            "reproducible": True,
        }
```

#### 2.1.3 Future Providers (Interface Examples)

```python
# For future migration

class OpenAIProvider(LLMProvider):
    """OpenAI API wrapper"""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation using openai package
        pass

class HuggingFaceProvider(LLMProvider):
    """HuggingFace Transformers wrapper"""
    def __init__(self, model: str = "mistralai/Mistral-7B"):
        self.model = model
        # Load from transformers pipeline
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation using transformers
        pass
```

---

## 3. Prompt Engineering Module

### 3.1 Template Strategy Pattern

Three prompt templates are implemented with increasing sophistication. Starts with role-play and migrates to chain-of-thought during development.

#### 3.1.1 Base Template Interface

```python
# src/rag/prompt_templates.py

from abc import ABC, abstractmethod
from typing import List, Dict

class PromptTemplate(ABC):
    """Abstract base for prompt generation strategies."""
    
    @abstractmethod
    def apply(self, query: str, documents: List[Dict]) -> str:
        """
        Generate a prompt from query and retrieved documents.
        
        Args:
            query: User question
            documents: Retrieved documents with metadata
            
        Returns:
            Formatted prompt ready for LLM
        """
        pass
    
    def _format_context(self, documents: List[Dict], max_chars: int = 3000) -> str:
        """Format documents into context string."""
        context = []
        total_chars = 0
        
        for doc in documents:
            doc_text = f"[{doc.get('id', 'unknown')}] {doc.get('title', 'Untitled')}\n"
            doc_text += f"{doc.get('content', '')[:500]}\n"
            
            if total_chars + len(doc_text) > max_chars:
                break
            
            context.append(doc_text)
            total_chars += len(doc_text)
        
        return "\n".join(context)
```

#### 3.1.2 Basic Template

```python
class BasicTemplate(PromptTemplate):
    """
    Simple template for testing.
    Trade-off: Low quality responses, minimal token usage.
    """
    
    def apply(self, query: str, documents: List[Dict]) -> str:
        context = self._format_context(documents)
        
        return f"""Context:
{context}

Question: {query}

Answer:"""
```

#### 3.1.3 Domain-Specific Role-Play Template

```python
class DomainSpecificTemplate(PromptTemplate):
    """
    Technical assistant role-play (INITIAL CHOICE).
    Trade-off: Good quality, moderate token usage, domain-aligned.
    Suitable for development and initial testing.
    """
    
    def __init__(self):
        self.system_prompt = """You are a technical assistant specialized in software and technology. 
Your role is to provide accurate, informative answers based on the provided documents.
Always cite your sources using [DOC_ID] format when referencing information."""
    
    def apply(self, query: str, documents: List[Dict]) -> str:
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
```

#### 3.1.4 Chain-of-Thought Template

```python
class ChainOfThoughtTemplate(PromptTemplate):
    """
    Chain-of-Thought reasoning.
    Trade-off: Best quality, higher token usage, explicit reasoning.
    Recommended for production and final evaluation.
    """
    
    def apply(self, query: str, documents: List[Dict]) -> str:
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
```

#### 3.1.5 Template Selection

```python
# src/rag/prompt_templates.py (continuation)

class PromptTemplateFactory:
    """Factory for creating prompt templates."""
    
    _templates = {
        "basic": BasicTemplate,
        "domain_specific": DomainSpecificTemplate,
        "chain_of_thought": ChainOfThoughtTemplate,
    }
    
    @classmethod
    def create(cls, template_type: str) -> PromptTemplate:
        """Create a prompt template by name."""
        if template_type not in cls._templates:
            raise ValueError(
                f"Unknown template: {template_type}. "
                f"Available: {list(cls._templates.keys())}"
            )
        return cls._templates[template_type]()
    
    @classmethod
    def available_templates(cls) -> List[str]:
        """List available template types."""
        return list(cls._templates.keys())
```

---

## 4. Citation Management

### 4.1 Citation Extraction

```python
# src/rag/citations.py

import re
from typing import List, Dict, Tuple

class CitationExtractor:
    """Extract and validate citations from LLM output."""
    
    @staticmethod
    def extract_citations(text: str, documents: List[Dict]) -> List[str]:
        """
        Extract citation IDs from text using [doc_id] format.
        
        Args:
            text: Generated text containing citations
            documents: Available documents for validation
            
        Returns:
            List of cited document IDs
        """
        # Pattern for [doc_id] citations
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        valid_doc_ids = {doc.get('id') for doc in documents}
        citations = [m for m in matches if m in valid_doc_ids]
        
        return list(dict.fromkeys(citations))  # Remove duplicates, preserve order
    
    @staticmethod
    def enrich_citations(
        citations: List[str],
        documents: List[Dict],
    ) -> List[Dict]:
        """
        Create rich citation objects with document metadata.
        
        Args:
            citations: List of document IDs
            documents: Available documents
            
        Returns:
            List of enriched citation dictionaries
        """
        doc_map = {doc.get('id'): doc for doc in documents}
        enriched = []
        
        for citation_id in citations:
            if citation_id in doc_map:
                doc = doc_map[citation_id]
                enriched.append({
                    "doc_id": citation_id,
                    "title": doc.get('title', 'Unknown'),
                    "url": doc.get('url', ''),
                    "source": doc.get('source', 'unknown'),
                    "snippet": doc.get('content', '')[:200],
                })
        
        return enriched
```

---

## 5. Output Parsing and Validation

### 5.1 JSON Structure Definition

```python
# src/rag/output_parser.py

from pydantic import BaseModel, Field
from typing import List, Optional
import json
import re

class Citation(BaseModel):
    """Citation object linking to source document."""
    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    url: Optional[str] = Field(None, description="Document URL")
    snippet: Optional[str] = Field(None, description="Relevant excerpt")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")

class RAGResponse(BaseModel):
    """Structured RAG response."""
    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Python is widely used for machine learning...",
                "citations": [
                    {
                        "doc_id": "doc_001",
                        "title": "Python for ML",
                        "url": "https://example.com",
                        "snippet": "Python has excellent libraries...",
                        "score": 0.95
                    }
                ]
            }
        }
```

### 5.2 Output Repair and Validation

```python
# src/rag/output_parser.py (continuation)

class OutputParser:
    """Parse and repair LLM output."""
    
    @staticmethod
    def repair_json(text: str) -> str:
        """
        Repair common JSON issues in LLM output.
        
        Handles:
        - Single quotes → double quotes
        - Trailing commas
        - Markdown code blocks
        - Common syntax errors
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*|\s*```', '', text)
        text = re.sub(r'^```', '', text)
        
        # Single to double quotes (careful with contractions)
        text = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', text)
        
        # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)
        text = re.sub(r',\s*}', '}', text)
        
        # Fix common quote issues
        text = text.replace('\'s', "'s")  # Possessives
        
        return text
    
    @classmethod
    def parse(cls, text: str) -> RAGResponse:
        """
        Parse LLM output to structured response.
        
        Implements fallback strategy:
        1. Try direct JSON parsing
        2. Try repair and retry
        3. Fall back to text extraction
        """
        # Attempt 1: Direct parsing
        try:
            data = json.loads(text)
            return RAGResponse(**data)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Repair and retry
        try:
            repaired = cls.repair_json(text)
            data = json.loads(repaired)
            return RAGResponse(**data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Attempt 3: Fallback extraction
        return cls._fallback_parse(text)
    
    @staticmethod
    def _fallback_parse(text: str) -> RAGResponse:
        """Fallback: extract answer and citations using regex."""
        # Try to find JSON answer block
        answer_match = re.search(
            r'"answer"\s*:\s*"([^"]*)"',
            text,
            re.DOTALL
        )
        answer = answer_match.group(1) if answer_match else text[:500]
        
        # Extract citations
        citations = re.findall(r'\[([^\]]+)\]', text)
        
        return RAGResponse(
            answer=answer,
            citations=[Citation(doc_id=c, title=c) for c in citations[:5]]
        )
```

---

## 6. RAG Module Implementation

### 6.1 Main RAG Orchestrator

```python
# src/rag/rag_module.py

from typing import List, Dict, Optional
from datetime import datetime
from .llm_provider import LLMProvider
from .prompt_templates import PromptTemplate, PromptTemplateFactory
from .citations import CitationExtractor
from .output_parser import OutputParser, RAGResponse
import logging

logger = logging.getLogger(__name__)

class RAGModule:
    """
    Retrieval-Augmented Generation module.
    
    Orchestrates the RAG pipeline:
    1. Query reception and validation
    2. Document retrieval (via SRIPipeline)
    3. Prompt construction
    4. LLM generation
    5. Citation extraction
    6. Response assembly
    """
    
    def __init__(
        self,
        pipeline,
        llm_provider: LLMProvider,
        template_type: str = "domain_specific",
        retrieval_top_k: int = 5,
        citation_threshold: float = 0.0,
    ):
        """
        Initialize RAG module.
        
        Args:
            pipeline: SRIPipeline instance
            llm_provider: LLM provider instance (Ollama, OpenAI, etc.)
            template_type: Prompt template strategy ("basic", "domain_specific", "chain_of_thought")
            retrieval_top_k: Number of documents to retrieve
            citation_threshold: Minimum relevance score for citations
        """
        self.pipeline = pipeline
        self.llm = llm_provider
        self.retrieval_top_k = retrieval_top_k
        self.citation_threshold = citation_threshold
        self.template = PromptTemplateFactory.create(template_type)
        self.citation_extractor = CitationExtractor()
        self.output_parser = OutputParser()
        
        # Validation
        if not self.llm.is_available():
            raise RuntimeError(
                f"LLM provider not available. "
            )
        
        logger.info(
            f"RAG Module initialized with {llm_provider.get_metadata()}. "
            f"Template: {template_type}"
        )
    
    def set_template(self, template_type: str) -> None:
        """Change prompt template strategy at runtime."""
        self.template = PromptTemplateFactory.create(template_type)
        logger.info(f"Prompt template changed to: {template_type}")
    
    def generate(
        self,
        query: str,
        documents: Optional[List[Dict]] = None,
        temperature: float = 0.7,
    ) -> RAGResponse:
        """
        Generate an augmented response for a query.
        
        Implements hybrid approach:
        - If documents=None: retrieves via pipeline
        - If documents provided: uses them directly
        
        Args:
            query: User question
            documents: Pre-filtered documents (optional)
            temperature: LLM creativity parameter
            
        Returns:
            RAGResponse with answer and citations
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Step 1: Document retrieval
        if documents is None:
            documents = self.pipeline.search(query, top_k=self.retrieval_top_k)
            if not documents:
                logger.warning("No documents retrieved. Returning error response.")
                return RAGResponse(
                    answer="No relevant documents found in the knowledge base. "
                           "Consider expanding the search or asking a different question.",
                    citations=[]
                )
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Step 2: Prompt construction
        prompt = self.template.apply(query, documents)
        logger.debug(f"Constructed prompt ({len(prompt)} chars)")
        
        # Step 3: LLM generation
        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=1024,
            )
            logger.info(f"LLM generated response ({len(raw_response)} chars)")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
        
        # Step 4: Parse structured response
        rag_response = self.output_parser.parse(raw_response)
        
        # Step 5: Enrich citations with document metadata
        citations = self.citation_extractor.extract_citations(
            rag_response.answer, documents
        )
        rag_response.citations = self.citation_extractor.enrich_citations(
            citations, documents
        )
        
        logger.info(
            f"Generated response with {len(rag_response.citations)} citations"
        )
        
        return rag_response
    
    def generate_batch(
        self,
        queries: List[str],
        temperature: float = 0.7,
    ) -> List[RAGResponse]:
        """
        Generate responses for multiple queries.
        
        Args:
            queries: List of questions
            temperature: LLM creativity parameter
            
        Returns:
            List of RAGResponse objects
        """
        responses = []
        for query in queries:
            try:
                response = self.generate(query, temperature=temperature)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                responses.append(
                    RAGResponse(
                        answer=f"Error processing query: {str(e)}",
                        citations=[]
                    )
                )
        
        return responses
```

---

## 7. Integration with SRIPipeline

### 7.1 Pipeline Extension

```python
# src/sri/pipeline.py (modifications)

from rag.rag_module import RAGModule
from rag.llm_provider import OllamaProvider

class SRIPipeline:
    """Extended to support RAG generation."""
    
    def __init__(self, ..., enable_rag: bool = False):
        # ... existing initialization ...
        
        self.rag_enabled = enable_rag
        self.rag_module = None
        
        if enable_rag:
            llm_provider = OllamaProvider(model="llama3.2:latest")
            self.rag_module = RAGModule(
                pipeline=self,
                llm_provider=llm_provider,
                template_type="domain_specific",  # Start with this
            )
    
    def search_with_generation(
        self,
        query: str,
        top_k: int = 10,
        with_rag: bool = True,
    ) -> dict:
        """
        Unified search interface supporting both retrieval and generation.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            with_rag: Whether to generate augmented answer
            
        Returns:
            {
                'retrieved_documents': [...],
                'generated_answer': "..." (if with_rag=True),
                'citations': [...] (if with_rag=True),
            }
        """
        # Retrieval
        documents = self.search(query, top_k=top_k)
        
        result = {
            'query': query,
            'retrieved_documents': documents,
            'document_count': len(documents),
        }
        
        # Generation (optional)
        if with_rag and self.rag_module:
            rag_response = self.rag_module.generate(
                query,
                documents=documents
            )
            result.update({
                'generated_answer': rag_response.answer,
                'citations': [c.dict() for c in rag_response.citations],
                'generation_timestamp': datetime.now().isoformat(),
            })
        
        return result
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/sri/rag/test_llm_provider.py

import pytest
from rag.llm_provider import OllamaProvider

class TestOllamaProvider:
    
    @pytest.fixture
    def provider(self):
        return OllamaProvider(model="llama3.2:latest")
    
    def test_initialization(self, provider):
        assert provider.model == "llama3.2:latest"
        assert provider.is_available()
    
    def test_generate(self, provider):
        response = provider.generate("What is Python?", temperature=0.5)
        assert isinstance(response, str)
        assert len(response) > 0

# tests/sri/rag/test_prompt_templates.py

def test_domain_specific_template():
    from rag.prompt_templates import DomainSpecificTemplate
    
    template = DomainSpecificTemplate()
    docs = [
        {"id": "doc1", "title": "Python Basics", "content": "Python is..."}
    ]
    prompt = template.apply("What is Python?", docs)
    
    assert "Python Basics" in prompt
    assert "What is Python?" in prompt

# tests/sri/rag/test_output_parser.py

def test_output_parser():
    from rag.output_parser import OutputParser, RAGResponse
    
    json_response = '{"answer": "Python", "citations": []}'
    parsed = OutputParser.parse(json_response)
    
    assert isinstance(parsed, RAGResponse)
    assert parsed.answer == "Python"
```

### 8.2 Integration Tests

```python
# tests/sri/rag/test_rag_integration.py

def test_rag_end_to_end(pipeline, llm_provider):
    """End-to-end RAG test."""
    from rag.rag_module import RAGModule
    
    rag = RAGModule(pipeline, llm_provider)
    response = rag.generate("How does LSI work?")
    
    assert response.answer
    assert isinstance(response.citations, list)
```

---

## 9. Implementation Roadmap

### 9.1 Phase 1: Foundation

**Deliverables:**
- [ ] LLM provider abstraction with Ollama implementation
- [ ] Prompt template framework with three strategies
- [ ] Citation extraction module
- [ ] Output parsing with JSON repair
- [ ] Unit tests for all components

**Code Files:**
- `src/rag/llm_provider.py`
- `src/rag/prompt_templates.py`
- `src/rag/citations.py`
- `src/rag/output_parser.py`
- `tests/sri/rag/`

### 9.2 Phase 2: Integration

**Deliverables:**
- [ ] RAGModule main orchestrator
- [ ] Integration with SRIPipeline
- [ ] Configuration management
- [ ] Logging and error handling
- [ ] Integration tests

**Code Files:**
- `src/rag/rag_module.py`
- `src/rag/__init__.py`
- Pipeline integration
- Configuration setup

### 9.3 Phase 3: Optimization

**Deliverables:**
- [ ] Template migration to chain-of-thought
- [ ] Performance optimization
- [ ] Fine-tuning and evaluation
- [ ] Documentation
- [ ] User guide

**Activities:**
- Benchmark different templates
- Cache optimization
- Prompt tuning for domain
- Evaluation against test set

---

## 10. Migration Paths

### 10.1 Future LLM Provider Migration

#### From Ollama to OpenAI

**Changes Required:**
1. Implement `OpenAIProvider` class
2. Add environment variable for API key
3. Single line change in initialization

```python
# Before
llm = OllamaProvider("llama3.2:latest")

# After
llm = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)
```

#### From Ollama to HuggingFace

**Changes Required:**
1. Implement `HuggingFaceProvider` class
2. Add model loading logic
3. Single line change in initialization

### 10.2 Prompt Architecture Migration

**Current (Query-Based):**
- Augments query with documents in prompt
- All context passed as text to LLM
- Single generation phase

#### Latent-Based RAG

- Pass document embeddings to LLM
- Requires LLM with embedding injection support
- Changes needed: PromptTemplate and RAGModule

#### Logits-Based RAG

- Access LLM internal representations
- Fuse document logits during decoding
- Requires deep LLM customization
- Requires LLM provider with logits hook support

---

## 11. Configuration Management

### 11.1 Environment Variables

```bash
# .env file

# Ollama Configuration
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300

# RAG Configuration
RAG_TEMPLATE=domain_specific  # or chain_of_thought
RAG_RETRIEVAL_TOP_K=5
RAG_TEMPERATURE=0.7
RAG_MAX_TOKENS=1024
RAG_CITATION_THRESHOLD=0.0

# Logging
RAG_LOG_LEVEL=INFO
```

### 11.2 Configuration Class

```python
# src/rag/config.py

from pydantic_settings import BaseSettings
from typing import Optional

class RAGConfig(BaseSettings):
    """RAG module configuration."""
    
    # LLM Configuration
    ollama_model: str = "llama3.2:latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300
    
    # RAG Configuration
    rag_template: str = "domain_specific"
    rag_retrieval_top_k: int = 5
    rag_temperature: float = 0.7
    rag_max_tokens: int = 1024
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

### 12. Test Queries for Evaluation

```python
test_queries = [
    "What are the main applications of Docker and Kubernetes?",
    "Explain the difference between React and Vue frameworks",
    "How do Large Language Models handle context?",
    "What are the best practices for Python machine learning?",
    "Describe the RAG architecture and its components",
]
```

---

## 13. Dependencies

### 13.1 New Dependencies Required

```toml
# pyproject.toml additions

[project]
dependencies = [
    # ... existing ...
    "requests>=2.31",  # For Ollama HTTP API
    "pydantic>=2.0",   # For structured outputs
]

[project.optional-dependencies]
rag = [
    "langchain>=0.2",  # Already listed
    "langchain-community>=0.2",  # Already listed
]
```

**Note:** All required packages are already in `pyproject.toml`.

---

## 14. Documentation and References

### 14.1 Bibliographic References

1. **RAG Architecture:**
   - Zhao, P., et al. (2024). "Retrieval-augmented generation for AI-generated content: A survey." *arXiv preprint* 2402.19473.
   - Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.

2. **LLM Evaluation:**
   - Es, S., & Schockaert, S. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv preprint* 2309.15217.

3. **Prompt Engineering:**
   - Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*.

### 14.2 Related Documentation

- Course Material: "Generación Aumentada por Recuperación (RAG)" - Lic. Carlos León González
- Project Guidelines: "Orientación_extracted.txt"
- System Architecture: "PRE_RAG_STATUS.md"

---

## 15. Risk Assessment and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Ollama service not running | Medium | High | Automatic is_available() check, clear error message |
| Low-quality LLM outputs | High | Medium | Start with domain_specific template, migrate to chain_of_thought |
| Hallucinations (LLM invents facts) | High | Medium | Enforce citation extraction, validate against documents |
| Citation extraction failures | Medium | Low | Implement fallback extraction, manual review |
| Token limit exceeded | Low | Medium | Document length limits, chunking strategy |
| Performance degradation | Low | Medium | Cache prompt templates, benchmark templates |

---

## 16. Success Metrics

- ✅ RAG module generates answers within 30 seconds (90th percentile)
- ✅ Citation accuracy ≥ 85% (manual evaluation on 20 sample outputs)
- ✅ Answer relevance rated ≥ 4/5 by evaluators
- ✅ Zero crashes on invalid inputs (graceful degradation)
- ✅ 100% test coverage on RAG core modules
- ✅ Documentation completeness: All public APIs documented in-code

---

## Appendix A: Quick Start Guide

```python
# Basic usage after implementation

from sri.pipeline import SRIPipeline
from rag.rag_module import RAGModule
from rag.llm_provider import OllamaProvider

# Initialize
pipeline = SRIPipeline()
llm = OllamaProvider(model="llama3.2:latest")
rag = RAGModule(pipeline, llm)

# Generate answer
response = rag.generate("How does LSI improve retrieval?")

# Access results
print(f"Answer: {response.answer}")
print(f"Citations: {response.citations}")

# Advanced: Custom documents
docs = pipeline.search("query", top_k=10)
response = rag.generate("query", documents=docs)

# Switch template at runtime
rag.set_template("chain_of_thought")
response = rag.generate("another query")
```

---