# Information Retrieval System with Retrieval-Augmented Generation

## Overview

The system is an end-to-end information retrieval and generation pipeline that integrates multiple independent components. The architecture provides unified orchestration of local search, sufficiency validation, conditional web search augmentation, and retrieval-augmented generation through the `MainOrchestrator` class.

---

## System Architecture

### Core Integration Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                   MainOrchestrator                              │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  query() → Unified Workflow:                                   │
│    1. Check database health                                    │
│    2. Auto-load from crawlers if empty                         │
│    3. Execute local search    [SRIPipeline]                   │
│           ↓                                                     │
│    4. Detect insufficiency    [3-criterion detection]         │
│           ↓                                                     │
│    5. If insufficient:        _search_web() [WebSearcher]     │
│           ↓                                                     │
│    6. Consolidate documents   [Deduplication]                 │
│           ↓                                                     │
│    7. Generate answer         [RAGModule + LLM]               │
│           ↓                                                     │
│    8. Return RAGResponse      [With metadata & citations]     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Independent Modules

### 1. SRIPipeline (Local Search)

**Location:** `src/sri/pipeline.py`

**Responsibility:** Multi-strategy local document retrieval

**Components:**
- LSI Model: Semantic latent factor extraction
- Vector Store: Embedding-based similarity search with ChromaDB support
- TF-IDF Index: Fallback term-frequency retrieval
- Ranking Engine: Multi-signal result ranking

**Input:** Query string  
**Output:** List of ranked documents with relevance scores

**Key Methods:**
- `search(query, top_k)` - Execute multi-strategy search
- `add_documents(documents)` - Index documents for retrieval
- `search_lsi(query)` - Semantic search via LSI
- `search_vector(query)` - Vector similarity search

---

### 2. CrawlerCaller (Data Acquisition)

**Location:** `src/sri/crawler/caller.py`

**Responsibility:** Orchestrate all six data crawlers, consolidate outputs, and manage document cache

**Spider Coordination:**
- DevTo crawler
- HackerNews crawler
- RealPython crawler
- Lobsters crawler
- TheNewStack crawler
- TheVerge crawler

**Input:** Configuration parameters (max_articles, force_recrawl)  
**Output:** Consolidated documents list

**Key Methods:**
- `execute_full_pipeline(force_recrawl, max_articles)` - Run all crawlers and consolidate
- `consolidate_raw_to_documents()` - Convert raw JSON to unified format
- `load_consolidated_documents()` - Load documents from cached documents.json file
- `merge_documents(new_documents)` - Merge new documents into cache without overwriting
- `clear_cached_documents()` - Remove data/raw and documents.json for complete reset
- `count_raw_documents()`, `count_consolidated_documents()` - Document counting

---

### 3. VectorStore (Embedding Backend)

**Location:** `src/retrieval/vector_store.py`

**Responsibility:** Manage embeddings and vector similarity search

**Capabilities:**
- Local in-memory backend with Python lists
- ChromaDB persistent backend (optional)
- Graceful fallback when ChromaDB unavailable

**Key Methods:**
- `add(ids, embeddings, metadatas, documents)` - Store embeddings
- `query(query_embeddings, n_results)` - Retrieve similar vectors
- `clear_all()` - Complete database cleanup including ChromaDB

---

### 4. InsufficientDetector (Quality Validation)

**Location:** `src/main_orchestator.py`

**Responsibility:** Apply 3-criterion insufficiency detection

**Detection Criteria:**
1. **Quantity:** Result count vs. MIN_RESULTS_FOR_QUERY
2. **Quality:** Average relevance score vs. MIN_AVG_SCORE_THRESHOLD
3. **Semantic:** Keyword overlap between query and results

**Input:** Query string and search results  
**Output:** Insufficiency assessment with reasons

**Key Methods:**
- `detect_insufficiency_for_query(query, results)` - Full assessment
- `_has_semantic_overlap(query, documents)` - Keyword matching

---

### 5. WebSearcher (Web Augmentation)

**Location:** `src/sri/web_search/searcher.py`

**Responsibility:** Retrieve documents from web using DuckDuckGo API

**Capabilities:**
- Web search via DuckDuckGo
- Result formatting to unified document schema
- Error handling with graceful fallback

**Input:** Query string  
**Output:** List of web search results

---

### 6. RAGModule (Answer Generation)

**Location:** `src/rag/rag_module.py`

**Responsibility:** Generate answers using local LLM with provided documents

**Components:**
- LLM Provider: Ollama interface
- Prompt Templates: Configurable generation strategies
- Output Parser: Response parsing and citation extraction

**Input:** Question, documents, configuration parameters  
**Output:** RAGResponse with answer and citations

**Key Methods:**
- `generate(question, documents, temperature, max_tokens)` - Generate answer

---

### 7. OutputParser (Response Formatting)

**Location:** `src/rag/output_parser.py`

**Responsibility:** Parse LLM output and extract citations

**Features:**
- Pydantic-based response validation
- Metadata tracking (auto_crawled, documents_used, timing)
- Citation extraction and linking

**Structures:**
- `RAGResponse`: Answer, citations, metadata
- Metadata fields: auto_crawled, insufficiency_detected, generation_time_seconds, timestamp

---

## MainOrchestrator Integration

### Class Definition

**Location:** `src/main_orchestator.py`

**Initialization:**
```python
class MainOrchestrator:
    def __init__(self):
        self.pipeline = SRIPipeline()
        self.llm_provider = LLMProvider()
        self.rag_module = RAGModule(llm_provider=self.llm_provider)
        self.sufficiency_checker = SufficiencyChecker()
        self.web_searcher = WebSearcher()
        self.crawler_caller = CrawlerCaller()
        self.settings = CrawlerSettings()
```

### Database Management Methods

**clear_all_indices() → Dict[str, Any]**
- Atomically clears VectorStore, ChromaDB collection, LSI models
- Removes cached crawler artifacts (data/raw and documents.json)
- Forces cold-start requiring web crawl on next load
- Returns success status with timestamp
- Error handling with logging

**load_documents_from_crawlers(max_articles, force_recrawl) → Dict[str, Any]**
- Attempts to load from cached documents.json if available
- Only executes crawlers if no consolidated file exists or force_recrawl=True
- Falls back to consolidating raw JSON to unified format
- Indexes documents in SRIPipeline
- Returns operation summary with document counts

**check_database_health() → Dict[str, Any]**
- Checks VectorStore document count
- Validates documents.json existence
- Detects ChromaDB availability
- Returns health status and actionable information

### Query Execution Pipeline

**query(question, max_local_results, enable_web_search, auto_reload) → RAGResponse**

**Execution Steps:**
1. Input validation (question cannot be empty)
2. Database health check
3. Auto-load from crawlers if database empty
4. Execute local search
5. Apply 3-criterion insufficiency detection
6. Conditional web search if insufficient
7. Persist web results to documents.json for future sessions (if found)
8. Document consolidation and deduplication
9. RAG-based answer generation
10. Metadata enrichment
11. Return complete RAGResponse

**Metadata Tracking:**
- auto_crawled: Whether crawlers were auto-executed
- total_documents_used: Final count of documents for RAG
- local_documents: Count of local search results
- web_documents: Count of web search results
- insufficiency_detected: Boolean flag
- insufficiency_reasons: List of detection criteria that triggered
- generation_time_seconds: Total execution time
- timestamp: ISO 8601 execution timestamp

**Web Result Persistence:**
- Web results are merged into documents.json after query completion
- Deduplication prevents duplicate IDs
- Indexing of web results deferred to next session (not re-indexed in current session)

### Supporting Methods

**_search_locally(query, max_results) → List[Dict]**
- Delegates to SRIPipeline.search()
- Returns ranked results with scores
- Logging on search completion

**_search_web(query, max_results) → List[Dict]**
- Delegates to WebSearcher.search()
- Returns web documents
- Error handling with warning logs

**_consolidate_documents(local_results, web_results) → List[Dict]**
- Deduplicates by content hash of first 100 characters
- Prioritizes local results over web results
- Returns consolidated unique document list

**_extract_content(document) → str**
- Utility method to extract content from multiple field names
- Supports: 'content', 'text', 'body' fields
- Returns empty string if no content found

**get_status() → Dict[str, Any]**
- Database status: health, document counts, ChromaDB availability
- Crawler status: raw document counts, consolidated counts
- Pipeline status: initialization state, LSI model availability
- Returns comprehensive system diagnostics

---

## Configuration

### Settings Location

**File:** `src/sri/crawler/settings.py`

**Crawler Thresholds:**
- `MIN_DOCUMENTS_THRESHOLD`: 50 - Minimum docs for sufficient database
- `MIN_AVG_SCORE_THRESHOLD`: 0.45 - Minimum average relevance score
- `MIN_RESULTS_FOR_QUERY`: 3 - Minimum results before web search
- `AUTO_CRAWL_ON_EMPTY`: True - Auto-execute crawlers if DB empty
- `AUTO_CRAWL_ON_INSUFFICIENT`: True - Auto-execute if results insufficient

**RAG Configuration File:** `src/rag/config.py`

**RAG Parameters:**
- `rag_template`: "domain_specific" - Template type for prompt generation
- `max_doc_content_length`: 1000 - Maximum characters of document content in prompts
- `max_snippet_length`: 200 - Maximum characters from each document in citations
- `rag_temperature`: 0.7 - LLM temperature for generation
- `rag_max_tokens`: 1024 - Maximum tokens in generated response

---

## Data Flow Example

### Scenario: Query with Auto-Load

**Input:** `query("How does LSI work?", auto_reload=True)`

**Step 1: Health Check**
- Database is empty (0 indexed documents)
- Auto-load enabled

**Step 2: Auto-Load**
- load_documents_from_crawlers() calls load_consolidated_documents()
- No cached documents.json found
- CrawlerCaller.execute_full_pipeline() starts
- All 6 crawlers execute sequentially
- Raw documents consolidated and saved to documents.json
- SRIPipeline.add_documents() indexes 150 new documents

**Step 3: Local Search**
- SRIPipeline.search("How does LSI work?", top_k=5)
- Returns 5 results: [0.82, 0.71, 0.65, 0.58, 0.42]

**Step 4: Insufficiency Detection**
- Count check: 5 ≥ 3 ✓
- Score check: average 0.636 ≥ 0.45 ✓
- Semantic check: keyword overlap detected ✓
- Result: Sufficient → no web search needed

**Step 5: RAG Generation**
- RAGModule.generate() with 5 local documents
- LLM generates answer with Ollama
- Citations extracted: references to top 2 documents

**Step 6: Response**
- RAGResponse returned with metadata
- metadata['auto_crawled'] = True
- metadata['total_documents_used'] = 5
- metadata['insufficiency_detected'] = False

---

## Error Handling and Recovery

### Component Failures

**SRIPipeline.search() fails:**
- Returns empty list
- Warning logged
- Sufficiency check receives 0 results
- Web search triggered if enabled

**WebSearcher.search() fails:**
- Error logged
- Returns empty list
- Proceeds with local results only

**RAGModule.generate() fails:**
- Error logged with full traceback
- RAGResponse returned with error message
- Metadata tracked with timing

**All failures:** Graceful degradation with meaningful error messages

---

## Execution Context

Execution context is determined by:
- LLM Provider: Ollama (http://localhost:11434)
- Consolidated cache: data/documents.json (loaded on startup; merged with web results)
- Raw crawler data: data/raw/{source}/*.json (only used if consolidated file missing)
- Indexed data: In-memory + optional ChromaDB at data/index/
- LSI models: Persisted to data/models/ (loaded on initialization)

**Database Lifecycle:**
1. **Startup:** Loads indices from data/index/ and data/models/ if available
2. **First Query:** If empty, loads from documents.json or auto-crawls
3. **Subsequent Queries:** Reuses in-memory indices (no re-crawl unless forced)
4. **Web Results:** Merged to documents.json at end of query (indexed next session)
5. **Reset:** `--clear-db` removes data/raw, documents.json, and indices (cold-start forced)

---

## Interface Agnostic Design

The `MainOrchestrator` API is designed for interface independence:

**Return Types:**
- All methods return structured `Dict` or Pydantic models
- No CLI string formatting in orchestrator logic
- Pure Python objects suitable for programmatic use

**Usage Patterns:**
- Direct instantiation for library use
- REST API wrapper for network access
- CLI wrapper for command-line interface
- Event-driven systems integration

---

## Summary

The system integrates seven independent specialized modules through the `MainOrchestrator` class:

| Module | Function | Interaction |
|--------|----------|-------------|
| SRIPipeline | Multi-strategy local search | Called by query execution |
| CrawlerCaller | Data acquisition, consolidation, and cache management | Auto-load from cache or crawl; merge web results |
| VectorStore | Embedding backend with ChromaDB | Part of SRIPipeline |
| InsufficientDetector | Quality validation | Decides web search trigger |
| WebSearcher | Web augmentation | Conditional execution; results persist to cache |
| RAGModule | Answer generation | Final stage of pipeline |
| OutputParser | Response formatting | Parses LLM output |

**Core interface:** `MainOrchestrator.query()` orchestrates the complete workflow from query input through RAG generation, with persistent document caching, automatic database management, intelligent insufficiency detection, and comprehensive metadata tracking.

**Cache Strategy:** documents.json acts as persistent seed for rapid startup; web results accumulate in cache across sessions without re-indexing until next explicit load.
