# Information Retrieval System with Retrieval-Augmented Generation (RAG)

**A Complete End-to-End Information Retrieval and Generation System for Technology and Software Domain**

## Executive Summary

This project implements a sophisticated information retrieval system that combines multiple retrieval strategies with Retrieval-Augmented Generation (RAG) to provide comprehensive, citation-backed answers to user queries. The system intelligently detects insufficient local search results using multi-criterion insufficiency detection and automatically augments the search with web results when necessary. Retrieved documents are then passed to a Large Language Model (LLM) via the RAG pipeline to generate natural language answers with extracted citations.

**Key Features:**
- 🔍 **Multi-Method Local Search**: LSI (Latent Semantic Indexing), TF-IDF, and vector similarity-based retrieval
- 🌐 **Intelligent Web Augmentation**: Automatic web search fallback with DuckDuckGo integration
- 🤖 **RAG-Based Answer Generation**: Leverages Ollama LLMs for citation-backed responses
- 📊 **Automatic Crawling**: Six specialized web crawlers for continuous document acquisition
- ✅ **Comprehensive Evaluation**: IR metrics (MAP, MRR, NDCG, Precision, Recall)
- 🎯 **Insufficiency Detection**: Quantity, quality, and semantic overlap criteria
- 📚 **Full Citation Tracking**: Automatic extraction and formatting of citations

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture and Components](#architecture-and-components)
3. [System Requirements](#system-requirements)
4. [Installation and Setup](#installation-and-setup)
5. [CLI Usage Guide](#cli-usage-guide)
6. [Backend Integration (MainOrchestrator)](#backend-integration-mainorchestrator)
7. [Configuration Reference](#configuration-reference)
8. [API Examples](#api-examples)
9. [Evaluation System](#evaluation-system)
10. [Troubleshooting](#troubleshooting)
11. [Team and Acknowledgments](#team-and-acknowledgments)
12. [Bibliography](#bibliography)
13. [License](#license)

---

## Project Structure

```
Information-Retrieval-System-Project/
├── src/                              # Source code
│   ├── main_orchestator.py           # Main orchestration layer (core API)
│   ├── main.py                       # Command-line interface
│   ├── rag_cli.py                    # RAG-specific CLI wrapper
│   │
│   ├── sri/                          # Core SRI (Sistema de Recuperación de Información)
│   │   ├── pipeline.py               # Main SRI pipeline orchestrator
│   │   ├── __init__.py
│   │   ├── crawler/                  # Web document acquisition
│   │   │   ├── base.py               # Base crawler class
│   │   │   ├── caller.py             # Crawler execution manager
│   │   │   ├── items.py              # Data models for crawler items
│   │   │   ├── pipeline.py           # Scrapy pipeline
│   │   │   ├── settings.py           # Crawler configuration
│   │   │   ├── spiders/              # 6 specialized web crawlers
│   │   │   │   ├── devto.py
│   │   │   │   ├── hackernews.py
│   │   │   │   ├── realpython.py
│   │   │   │   ├── lobsters.py
│   │   │   │   ├── thenewstack.py
│   │   │   │   └── theverge.py
│   │   │   └── __main__.py
│   │   │
│   │   └── web_search/               # Web search augmentation
│   │       ├── checker.py            # Insufficiency detection logic
│   │       ├── searcher.py           # DuckDuckGo web search
│   │       ├── indexer.py            # Web result processing
│   │       └── pipeline.py           # Web search pipeline
│   │
│   ├── rag/                          # Retrieval-Augmented Generation
│   │   ├── rag_module.py             # Main RAG orchestrator
│   │   ├── llm_provider.py           # LLM provider abstraction (Ollama)
│   │   ├── config.py                 # RAG configuration
│   │   ├── prompt_templates.py       # Prompt engineering templates
│   │   ├── citations.py              # Citation extraction and formatting
│   │   ├── output_parser.py          # RAG response parsing
│   │   └── __init__.py
│   │
│   ├── retrieval/                    # Document retrieval methods
│   │   ├── lsi_model.py              # Latent Semantic Indexing
│   │   ├── vector_store.py           # Vector storage (ChromaDB/custom backend)
│   │   └── __pycache__/
│   │
│   ├── indexing/                     # Text indexing
│   │   ├── indexer.py                # Inverted index and TF-IDF
│   │   └── __pycache__/
│   │
│   ├── ranking/                      # Result ranking
│   │   ├── ranking.py                # Multi-signal ranking engine
│   │   └── __pycache__/
│   │
│   ├── evaluation/                   # System evaluation
│   │   ├── evaluation.py             # IR metrics computation
│   │   └── __pycache__/
│   │
│   ├── app/                          # Frontend application
│   └── acquisition/                  # Placeholder for future data acquisition
│
├── data/                             # Data storage
│   ├── documents.json                # Consolidated document collection
│   ├── evaluation/                   # Evaluation results
│   ├── index/                        # Vector store and indices
│   │   └── chroma.sqlite3            # ChromaDB persistence
│   ├── processed/                    # Processed documents
│   └── qrels/                        # Relevance judgments
│
├── tests/                            # Test suite
│   ├── sri/
│   │   ├── test_system.py            # System integration tests
│   │   ├── crawler/                  # Crawler tests
│   │   ├── indexer/                  # Indexing tests
│   │   ├── ranking/                  # Ranking tests
│   │   ├── retrieval/                # Retrieval tests
│   │   ├── vectordb/                 # Vector store tests
│   │   ├── rag/                      # RAG tests
│   │   └── web_search/               # Web search tests
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE_ANALYSIS.md      # Detailed architecture analysis
│   ├── FRONTEND_IMPLEMENTATION_PLAN.md  # Gradio interface specification
│   ├── PRE_RAG_STATUS.md             # Pre-RAG implementation status
│   └── RAG_IMPLEMENTATION_PLAN.md    # RAG implementation details
│
├── pyproject.toml                    # Project metadata and dependencies
├── Makefile                          # Build and development commands
├── docker-compose.yml                # Docker orchestration
├── Dockerfile                        # Container specification
└── README.md                         # This file
```

### Key Directory Functions

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `src/` | All source code | Python modules and CLI entry points |
| `src/sri/` | Core retrieval system | Pipeline, crawlers, web search |
| `src/rag/` | Answer generation | LLM integration, prompts, citations |
| `src/retrieval/` | Document retrieval | LSI, vector storage, embeddings |
| `data/` | Persistent data storage | Documents, indices, evaluation results |
| `tests/` | Test suite | Unit and integration tests |
| `docs/` | Project documentation | Architecture, implementation plans |

---

## Architecture and Components

### 1. System Overview

The system operates in the following pipeline:

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│ MainOrchestrator (main_orchestator.py)      │
│ ┌──────────────────────────────────────┐   │
│ │ 1. Database Health Check            │   │
│ │    • Check indexed document count   │   │
│ │    • Verify vector store availability│   │
│ └──────────────────────────────────────┘   │
│           ↓                                  │
│ ┌──────────────────────────────────────┐   │
│ │ 2. Local Search (SRIPipeline)        │   │
│ │    • LSI (semantic similarity)       │   │
│ │    • TF-IDF (term frequency)         │   │
│ │    • Vector similarity (embeddings)  │   │
│ └──────────────────────────────────────┘   │
│           ↓                                  │
│ ┌──────────────────────────────────────┐   │
│ │ 3. Insufficiency Detection           │   │
│ │    • Quantity: Few results?          │   │
│ │    • Quality: Low relevance scores?  │   │
│ │    • Semantic: Keyword overlap?      │   │
│ └──────────────────────────────────────┘   │
│           ↓                                  │
│ ┌──────────────────────────────────────┐   │
│ │ 4. Conditional Web Search (Optional) │   │
│ │    • DuckDuckGo augmentation         │   │
│ │    • Persist results for future use  │   │
│ └──────────────────────────────────────┘   │
│           ↓                                  │
│ ┌──────────────────────────────────────┐   │
│ │ 5. Document Consolidation            │   │
│ │    • Merge local + web results       │   │
│ │    • Deduplication                   │   │
│ └──────────────────────────────────────┘   │
│           ↓                                  │
│ ┌──────────────────────────────────────┐   │
│ │ 6. RAG Generation (RAGModule)        │   │
│ │    • Prompt template application     │   │
│ │    • Ollama LLM inference            │   │
│ │    • Citation extraction             │   │
│ └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
    ↓
RAGResponse (answer + citations)
```

### 2. Core Components

#### MainOrchestrator (`src/main_orchestator.py`)

**Central API for the entire system.** Orchestrates all operations and exposes a clean interface independent of the front-end technology (CLI, GUI, REST API, etc.).

**Key Methods:**

| Method | Purpose | Parameters | Returns |
|--------|---------|-----------|---------|
| `query()` | Complete end-to-end query pipeline | `question, max_local_results, enable_web_search, auto_reload` | `RAGResponse` |
| `retrieve_documents()` | Document retrieval without RAG | Same as `query()` | `Dict[str, Any]` with documents |
| `augment_response()` | RAG generation from documents | `question, documents` | `RAGResponse` |
| `clear_all_indices()` | Clear database | None | `Dict[str, Any]` status |
| `load_documents_from_crawlers()` | Execute crawlers and index | `max_articles, force_recrawl` | `Dict[str, Any]` statistics |
| `check_database_health()` | Database readiness check | None | `Dict[str, Any]` health metrics |
| `detect_insufficiency_for_query()` | Insufficiency detection | `query, results` | `Dict[str, Any]` assessment |
| `evaluate_test()` | System evaluation | `test_spec` (optional) | `Dict[str, Any]` IR metrics |
| `get_status()` | System diagnostics | None | `Dict[str, Any]` complete status |

**Internal Architecture:**
- **Database Management**: Manages VectorStore, ChromaDB, LSI models, and consolidated documents
- **Search Pipeline**: Orchestrates local search via SRIPipeline
- **Insufficiency Detection**: Multi-criterion analysis (quantity, quality, semantic)
- **Web Augmentation**: Conditional DuckDuckGo integration
- **RAG Integration**: Passes documents to RAGModule for generation
- **Crawlers**: Manages six specialized web crawlers for document acquisition

#### SRIPipeline (`src/sri/pipeline.py`)

**Unified interface for local document search.**

**Methods:**
- `index(documents, save=True)` — Index documents using LSI, TF-IDF, and vector storage
- `search(query, top_k=10)` — Retrieve relevant documents using multi-method scoring
- `evaluate(test_set)` — Compute IR evaluation metrics

**Integrated Components:**
- **InvertedIndex** (`src/indexing/indexer.py`) — Vocabulary, posting lists, TF-IDF
- **LSIModel** (`src/retrieval/lsi_model.py`) — Latent semantic analysis
- **VectorStore** (`src/retrieval/vector_store.py`) — ChromaDB or custom backend
- **RankingEngine** (`src/ranking/ranking.py`) — Multi-signal result ranking

#### RAGModule (`src/rag/rag_module.py`)

**Orchestrates the Retrieval-Augmented Generation pipeline.**

**Methods:**
- `generate(query, documents, temperature, max_tokens)` — Generate answer with citations

**Components:**
- **LLMProvider** (`src/rag/llm_provider.py`) — Ollama interface
- **PromptTemplateFactory** (`src/rag/prompt_templates.py`) — Template selection (basic, domain_specific, chain_of_thought)
- **CitationExtractor** (`src/rag/citations.py`) — Automatic citation extraction
- **OutputParser** (`src/rag/output_parser.py`) — Response structuring

#### Web Search and Insufficiency Detection (`src/sri/web_search/`)

**Automatic augmentation of local results with web search.**

- **SufficiencyChecker** (`checker.py`) — Multi-criterion insufficiency detection
- **WebSearcher** (`searcher.py`) — DuckDuckGo integration
- **WebIndexer** (`indexer.py`) — Web result processing and formatting

#### Web Crawlers (`src/sri/crawler/`)

**Six specialized crawlers for continuous document acquisition:**

1. **DevTo** — Technology articles and tutorials
2. **HackerNews** — Tech news and discussions
3. **RealPython** — Python programming resources
4. **Lobsters** — Software engineering news
5. **TheNewStack** — Cloud-native and DevOps content
6. **TheVerge** — Tech product reviews and news

**Orchestrated by:**
- **CrawlerCaller** (`caller.py`) — Execution manager
- **CrawlerSettings** (`settings.py`) — Configuration

---

## System Requirements

### Python Environment
- **Python**: 3.10 or higher
- **Package Manager**: pip or uv (recommended)

### External Dependencies

#### Runtime Requirements
- **Ollama**: For LLM inference (required for RAG)
  - Default URL: `http://localhost:11434`
  - Default Model: `llama3.2:latest` (customizable)
  - Installation: https://ollama.ai

#### Core Libraries
See `pyproject.toml` for complete dependency list.

**Critical Dependencies:**
- `scikit-learn>=1.6` — LSI, TF-IDF, machine learning
- `chromadb>=0.5` — Vector database with embeddings
- `sentence-transformers>=2.7` — Text embeddings
- `langchain>=0.2` — LLM integration framework
- `duckduckgo-search>=6.0` — Web search fallback
- `scrapy>=2.11` — Web crawling framework
- `nltk>=3.8` — Natural language processing
- `gradio>=4.0` — Web interface framework

### System Resources
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB+ for document indexing and vector store
- **Network**: Required for crawler and web search operations

---

## Installation and Setup

### 1. Clone Repository

```bash
git clone https://github.com/D4R102004/Information-Retrieval-System-Project.git
cd Information-Retrieval-System-Project
```

### 2. Create Virtual Environment

```bash
# Using Python venv
python3.10 -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Using pip
pip install -e ".[dev]"

# OR using uv (faster, recommended)
uv pip install -e ".[dev]"
```

### 4. Install Ollama

```bash
# Visit https://ollama.ai and download the installer for your OS

# Verify installation
ollama pull llama3.2:latest
ollama serve  # Start Ollama service in background
```

### 5. Initialize Pre-commit Hooks (Optional)

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

### 6. Verify Installation

```bash
python -c "from main_orchestator import MainOrchestator; print('[OK] MainOrchestrator imported successfully')"
```

---

## CLI Usage Guide

### Command Syntax

The CLI is accessed via `src/main.py` with the following structure:

```bash
python src/main.py [MODE] [OPTIONS]
```

### Available Modes

#### 1. Single Query Execution

Execute a single query and display results:

```bash
python src/main.py --query "How does machine learning work?"
```

**Output includes:**
- Formatted answer with proper markdown rendering
- Citation list with sources and snippets
- Metadata (local documents, web documents, generation time)

**With custom parameters:**

```bash
# Use more local results
python src/main.py --query "Python async programming" --max-local 10

# Disable web search augmentation
python src/main.py --query "Your question" --no-web-search

# Increase verbosity for debugging
python src/main.py --query "Your question" -v

# Write logs to file
python src/main.py --query "Your question" --log-file system.log
```

#### 2. Interactive Mode

Enter a loop where you can ask multiple queries sequentially:

```bash
python src/main.py --interactive
```

**Interactive commands:**
```
ask <query>      - Submit a query
status           - Display database health
load             - Load documents from crawlers
clear            - Clear database (with confirmation)
help             - Show help message
exit             - Exit application
```

**Example session:**
```
ask What is LSI?
ask How does ChromaDB work?
status
load
exit
```

#### 3. Database Status

Display current database health and statistics:

```bash
python src/main.py --status
```

**Output:**
```
========================================================
DATABASE STATUS
========================================================
Status:                healthy / degraded / empty
Indexed Documents:     1250 (Min required: 500)
File Documents:        1250 (data/documents.json)
Raw Documents:         0 (data/raw/)
ChromaDB Available:    Yes
VectorStore Count:     1250
LLM Connection:        ✓ Connected (latency: 45ms)
Last Update:           2025-05-31 14:23:15 UTC
========================================================
```

#### 4. Load Data from Crawlers

Execute the full crawl → consolidate → index pipeline:

```bash
# Standard load (respects existing crawls)
python src/main.py --load-data

# Force recrawl (ignore cache)
python src/main.py --load-data --force

# Load with custom limits
python src/main.py --load-data --max-articles 500
```

**Output:**
- Crawler execution status for each of 6 spiders
- Consolidated document count
- Indexing progress
- Final statistics

#### 5. Clear Database

Remove all indices, models, and documents:

```bash
# With confirmation prompt
python src/main.py --clear-db

# Force clear without confirmation
python src/main.py --clear-db --force
```

**Warning:** This operation cannot be undone.

### Global Options

```
--verbose, -v              Enable debug logging
--log-file PATH            Write logs to file
--force                    Skip confirmations
--max-articles N           Max articles per crawler (default: 100)
--max-local N              Max local search results (default: 5)
--no-web-search            Disable web search fallback
```

### Complete Examples

**Research query with full output:**
```bash
python src/main.py --query "Distributed systems consensus algorithms" \
                   --max-local 10 \
                   --verbose \
                   --log-file research.log
```

**Bulk load and evaluate:**
```bash
python src/main.py --load-data --max-articles 1000 --verbose
```

**Interactive session with logging:**
```bash
python src/main.py --interactive --log-file session.log --verbose
```

---

## Backend Integration (MainOrchestrator)

### Direct Python API Usage

The MainOrchestrator can be used directly in Python code without the CLI:

```python
from main_orchestator import MainOrchestrator

# Initialize
orchestrator = MainOrchestrator()

# Single query (complete pipeline)
response = orchestrator.query(
    question="What is Retrieval-Augmented Generation?",
    max_local_results=5,
    enable_web_search=True,
    auto_reload=True
)

print(f"Answer: {response.answer}")
print(f"Citations: {len(response.citations)}")
print(f"Metadata: {response.metadata}")
```

### Advanced Usage Examples

#### 1. Document Retrieval Without RAG

```python
# Get documents without generating an answer
result = orchestrator.retrieve_documents(
    question="machine learning frameworks",
    max_local_results=10,
    enable_web_search=True,
    auto_reload=False
)

documents = result['documents']
metadata = result['metadata']

print(f"Local docs: {metadata['local_documents']}")
print(f"Web docs: {metadata['web_documents']}")
print(f"Insufficiency: {metadata['insufficiency_reasons']}")
```

#### 2. RAG Generation from Custom Documents

```python
# Generate answer from pre-selected documents
documents = [
    {
        "id": "doc1",
        "title": "Deep Learning Fundamentals",
        "content": "Neural networks are...",
        "url": "https://example.com/article1"
    },
    # ... more documents
]

response = orchestrator.augment_response(
    question="Explain backpropagation",
    documents=documents
)

print(response.answer)
for citation in response.citations:
    print(f"  - {citation.title}")
```

#### 3. Multi-Step Query with Custom Parameters

```python
# Step 1: Retrieve and check insufficiency
retrieval = orchestrator.retrieve_documents(
    question="query",
    max_local_results=3,
    enable_web_search=False  # Don't use web search in this step
)

# Step 2: Check insufficiency
insufficiency = orchestrator.detect_insufficiency_for_query(
    query="query",
    results=retrieval['documents']
)

if insufficiency['is_insufficient']:
    print(f"Reasons: {insufficiency['reasons']}")
    # Optionally refine query or fetch more documents

# Step 3: Generate answer
response = orchestrator.augment_response(
    question="query",
    documents=retrieval['documents']
)
```

#### 4. System Diagnostics

```python
# Complete system status
status = orchestrator.get_status()
print(f"Database status: {status['database']['status']}")
print(f"Indexed docs: {status['database']['indexed_documents']}")
print(f"Raw docs: {status['crawlers']['raw_documents']}")

# Database health
health = orchestrator.check_database_health()
print(f"Ready to search: {health['can_search']}")
```

#### 5. Database Operations

```python
# Clear all data
result = orchestrator.clear_all_indices()
if result['success']:
    print("Database cleared")

# Load data from crawlers
load_result = orchestrator.load_documents_from_crawlers(
    max_articles=1000,
    force_recrawl=False
)
print(f"Indexed: {load_result['indexed_documents']} documents")
```

---

## Configuration Reference

### RAG Configuration (`src/rag/config.py`)

```python
from rag.config import config

# View current configuration
print(config.ollama_model)      # Current model: "llama3.2:latest"
print(config.temperature)       # Generation randomness: 0.7
print(config.max_tokens)        # Max response length: 1024

# Modify configuration (at runtime)
config.temperature = 0.5
config.max_tokens = 2048
```

**Key Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `ollama_model` | str | `llama3.2:latest` | - | Model identifier for inference |
| `ollama_base_url` | str | `http://localhost:11434` | - | Ollama service URL |
| `rag_template` | str | `domain_specific` | basic, domain_specific, chain_of_thought | Prompt template strategy |
| `temperature` | float | `0.7` | 0.0–1.0 | Response randomness (0=deterministic, 1=random) |
| `max_tokens` | int | `1024` | 100–4096 | Maximum response length in tokens |
| `max_cites` | int | `10` | 1–20 | Maximum citations to extract |
| `top_k_retrieval` | int | `5` | 1–20 | Documents to pass to RAG |

### Crawler Configuration (`src/sri/crawler/settings.py`)

```python
from sri.crawler.settings import CrawlerSettings

settings = CrawlerSettings()
print(settings.MIN_RESULTS_FOR_QUERY)      # Minimum results threshold: 5
print(settings.MIN_AVG_SCORE_THRESHOLD)    # Minimum avg relevance: 0.3
```

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MIN_RESULTS_FOR_QUERY` | int | 5 | Minimum local results before web search |
| `MIN_AVG_SCORE_THRESHOLD` | float | 0.3 | Minimum average relevance score |
| `MIN_ARTICLES_PER_SPIDER` | int | 50 | Minimum articles per crawler |

---

## API Examples

### REST API (Future Integration)

The MainOrchestrator API is designed for REST API wrapping:

```python
# Pseudocode for FastAPI wrapper
from fastapi import FastAPI
from main_orchestator import MainOrchestrator

app = FastAPI()
orchestrator = MainOrchestrator()

@app.post("/api/query")
async def query_endpoint(question: str, max_local: int = 5):
    response = orchestrator.query(
        question=question,
        max_local_results=max_local
    )
    return response.dict()

@app.get("/api/status")
async def status_endpoint():
    return orchestrator.get_status()
```

### Python SDK Usage

```python
# As a Python library
from main_orchestator import MainOrchestrator

class SearchClient:
    def __init__(self):
        self.orchestrator = MainOrchestrator()
    
    def search(self, question: str):
        return self.orchestrator.query(question)
    
    def evaluate(self, test_file: str):
        with open(test_file) as f:
            test_spec = json.load(f)
        return self.orchestrator.evaluate_test(test_spec)

client = SearchClient()
response = client.search("Your question")
```

---

## Evaluation System

### Running Evaluations

**Via CLI:**
```bash
python src/main.py --evaluate
```

**Via Python API:**
```python
# Run evaluation with default test set (data/test_queries.json)
results = orchestrator.evaluate_test()

# Run evaluation with custom test set
custom_tests = {
    "test_queries": [
        {
            "query_id": "q1",
            "query": "machine learning",
            "relevant": ["doc1", "doc2"],
            "grades": {"doc1": 3, "doc2": 2}
        },
        # ... more queries
    ]
}
results = orchestrator.evaluate_test(custom_tests)
```

### Evaluation Metrics

The system computes standard Information Retrieval metrics:

**Per-Query Metrics:**
- **AP** (Average Precision) — Area under precision-recall curve
- **RR** (Reciprocal Rank) — Position of first relevant document
- **P@k** (Precision at k) — Proportion of relevant in top-k
- **R@k** (Recall at k) — Proportion of relevant retrieved in top-k
- **NDCG@k** (Normalized DCG) — Ranking quality with graded relevance

**Aggregate Metrics:**
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **Mean P@1, @3, @5, @10**
- **Mean R@1, @3, @5, @10**
- **Mean NDCG@1, @3, @5, @10**

**Evaluation Output:**
```json
{
    "status": "success",
    "timestamp": "2025-05-31T14:30:00Z",
    "execution_time_seconds": 45.3,
    "aggregate": {
        "num_queries": 10,
        "MAP": 0.652,
        "MRR": 0.847,
        "mean_P@1": 0.7,
        "mean_R@5": 0.432,
        "mean_NDCG@5": 0.715
    },
    "per_query": [
        {
            "query_id": "q1",
            "num_relevant": 5,
            "ap": 0.85,
            "rr": 1.0,
            "p@5": 0.8,
            "r@5": 0.4,
            "ndcg@5": 0.895
        },
        // ... more queries
    ]
}
```

### Test File Format

**data/test_queries.json:**
```json
{
    "test_queries": [
        {
            "query_id": "q1",
            "query": "search query text",
            "relevant": ["doc_id1", "doc_id2", "doc_id3"],
            "grades": {
                "doc_id1": 3,
                "doc_id2": 2,
                "doc_id3": 1
            }
        },
        // ... more test queries
    ]
}
```

---

## Troubleshooting

### Common Issues

#### 1. Database Empty Error

**Problem:** "Database is empty. A search is impossible to perform"

**Solutions:**
```bash
# Load data from crawlers
python src/main.py --load-data

# Or via Python API
orchestrator.load_documents_from_crawlers()
```

**Check minimum documents:**
```bash
python src/main.py --status
```

Ensure `Indexed Documents` ≥ 500 (configurable `self.settings["min_documents"]`).

#### 2. Ollama Connection Failed

**Problem:** "Failed to connect to Ollama service"

**Solutions:**
```bash
# 1. Start Ollama service
ollama serve

# 2. Verify installation
ollama list

# 3. Pull required model
ollama pull llama3.2:latest

# 4. Check URL configuration
# Default: http://localhost:11434
# Modify in src/rag/config.py if different
```

#### 3. Low RAG Quality / Weak Citations

**Problem:** Generated answers lack relevant citations

**Solutions:**
1. **Increase local search results:**
   ```bash
   python src/main.py --query "Your question" --max-local 10
   ```

2. **Verify document relevance:**
   ```bash
   python src/main.py --status
   ```

3. **Adjust RAG parameters:**
   ```python
   from rag.config import config
   config.temperature = 0.3  # More deterministic
   config.max_tokens = 2048  # Longer responses
   ```

4. **Use different prompt template:**
   ```python
   rag = RAGModule(llm, template_type="chain_of_thought")
   ```

#### 4. Crawler Failures

**Problem:** "Crawler execution failed"

**Solutions:**
```bash
# Check network connectivity
ping duckduckgo.com

# Force recrawl
python src/main.py --load-data --force

# Increase timeout
python src/main.py --load-data --verbose
```

#### 5. Memory Issues During Indexing

**Problem:** Out of memory when loading large datasets

**Solutions:**
1. Reduce documents per crawler:
   ```bash
   python src/main.py --load-data --max-articles 100
   ```

2. Use ChromaDB (more memory-efficient):
   - Install: `pip install chromadb`
   - System will auto-detect and use it

3. Increase system RAM or use pagination in custom code

---

## Development and Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test module
pytest tests/sri/test_system.py -v

# Run with coverage
pytest --cov=src tests/

# Run specific test function
pytest tests/sri/retrieval/test_lsi_model.py::test_lsi_fitting -v
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Full check
make check
```

### Development Commands

```bash
# View all available commands
make help

# Install development dependencies
make install

# Clean cache files
make clean
```

---

## Team and Acknowledgments

**Project Members:**
- **Darío Francisco Alfonso** (@D4R102004)
- **Juan Carlos Carmenate** (@Juank404)
- **Sebastian González Alfonso** (@sebagonz106)

**Advisors:**
- University of Havana, Faculty of Mathematics and Computer Science

**Contributions:**
We acknowledge the open-source community for the excellent libraries that power this system: scikit-learn, Ollama, ChromaDB, Scrapy, and many others.

---

## Bibliography

### Core References

- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6), 391–407.

### Retrieval-Augmented Generation

- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *arXiv preprint arXiv:2005.11401*.

### Vector Databases and Embeddings

- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. (2019). *arXiv preprint arXiv:1908.10084*.

### Information Retrieval Evaluation

- Baeza-Yates, R., & Ribeiro-Neto, B. (2011). *Modern information retrieval: The concepts and technology behind search* (2nd ed.). Addison-Wesley.

### Web Crawling and Data Acquisition

- Scrapy Documentation. https://docs.scrapy.org/

---

## License

**Academic Project**  
Universidad de La Habana, 2025–2026

This project is developed as an academic exercise in Information Retrieval systems and Natural Language Processing. All code is provided as-is for educational purposes.

---

## Quick Start

**Get started in 5 minutes:**

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Start Ollama
ollama pull llama3.2:latest && ollama serve

# 3. Load data
python src/main.py --load-data

# 4. Ask a question
python src/main.py --query "What is machine learning?"

# 5. Try interactive mode
python src/main.py --interactive
```

---

**For detailed architecture documentation, see:**

- `docs/INDEX.md` — Documentation structure guide
- `docs/ARCHITECTURE_ANALYSIS.md` — Technical deep-dive

## Optional recommendation module

The project now includes a content-based recommendation module under `src/recommendation/`.
It recommends documents using a hybrid score composed of:

- TF-IDF content similarity over title, tags, source, and article content.
- Optional user interests or current query text.
- Optional liked/seed document IDs.
- Freshness and source-prior signals.

Main backend methods exposed through `MainOrchestator`:

```python
orchestrator.recommend_documents(
    query="serverless websocket apps",
    interests="cloud computing, javascript, APIs",
    liked_doc_ids=["010a7286-edfa-4143-9e46-462829787546"],
    top_k=10,
)

orchestrator.recommend_similar_documents(
    document_id="010a7286-edfa-4143-9e46-462829787546",
    top_k=10,
)
```

The Gradio UI also includes a new **Recommendation** tab where the user can generate
profile-based recommendations or find documents similar to a selected document ID.
