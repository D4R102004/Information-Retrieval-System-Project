# Backend Architecture Deep-Dive

**Comprehensive Technical Analysis of the Information Retrieval and RAG System**

*This document provides detailed architectural analysis of the MainOrchestrator and supporting components, intended for developers and system maintainers.*

---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [MainOrchestrator Architecture](#mainorchestrator-architecture)
3. [Component Interactions](#component-interactions)
4. [Data Flow Pipelines](#data-flow-pipelines)
5. [Database Architecture](#database-architecture)
6. [Search Methodology](#search-methodology)
7. [RAG Pipeline](#rag-pipeline)
8. [Error Handling and Resilience](#error-handling-and-resilience)
9. [Performance Considerations](#performance-considerations)
10. [Extension Points](#extension-points)

---

## Executive Overview

The backend system is organized around a **central orchestration pattern** where `MainOrchestator` acts as the unified interface to all subsystems. This design enables:

- **Interface Independence**: The API is completely decoupled from CLI, GUI, or REST API implementations
- **Stateless Operations**: Each method can be called independently without side effects
- **Atomic Transactions**: Database operations maintain consistency
- **Gradual Enhancement**: Documents and results are enriched at each pipeline stage
- **Comprehensive Logging**: All operations are traced for debugging and audit trails

---

## MainOrchestator Architecture

### Class Hierarchy and Initialization

```python
class MainOrchestator:
    def __init__(self):
        # Component Initialization (all singletons)
        self.pipeline = SRIPipeline()
        self.llm_provider = OllamaProvider()
        self.rag_module = RAGModule(llm=self.llm_provider)
        self.sufficiency_checker = SufficiencyChecker()
        self.web_searcher = WebSearcher()
        self.crawler_caller = CrawlerCaller()
        self.settings = CrawlerSettings()
        
        # File system paths
        self.data_dir = Path(...) / "data"
        self.documents_path = self.data_dir / "documents.json"
        self.raw_data_dir = self.data_dir / "raw"
```

**Key Design Decisions:**

1. **Single Initialization**: All components are initialized once and reused across requests
2. **Shared Pipeline**: Multiple concurrent queries use the same `SRIPipeline` instance (thread-safe design is implicit)
3. **Path Management**: All file paths are resolved relative to the project root for consistency
4. **Lazy Loading**: Heavy components (LLM, models) are only instantiated when needed

### Method Organization

The class methods are organized into logical sections:

#### Section 1: Database Management

**Responsibility**: Handle database state, clear operations, and health checks

```python
def clear_all_indices(self) -> Dict[str, Any]:
    """Atomically clear all indices, models, and documents."""

def load_documents_from_crawlers(
    self,
    max_articles: int = 1000,
    force_recrawl: bool = False
) -> Dict[str, Any]:
    """Execute crawl → consolidate → index pipeline."""

def check_database_health(self) -> Dict[str, Any]:
    """Check database status and readiness."""
```

**State Transitions:**

```
[Empty DB]
    ↓
[User calls auto_reload_empty=True]
    ↓
[Attempts: consolidated file → raw consolidation → crawlers]
    ↓
[Success: MIN_DB_DOCUMENTS indexed] OR [Failure]
```

#### Section 2: Insufficiency Detection

**Responsibility**: Multi-criterion analysis to determine if web search is needed

```python
def _has_semantic_overlap(self, query: str, documents: List[Dict]) -> bool:
    """Detect semantic overlap between query keywords and documents."""

def detect_insufficiency_for_query(
    self,
    query: str,
    results: List[Dict]
) -> Dict[str, Any]:
    """Apply 3-criterion insufficiency detection."""
```

**Three Criteria:**

1. **Quantity**: `len(results) < MIN_RESULTS_FOR_QUERY` (default: 5)
2. **Quality**: `avg_score < MIN_AVG_SCORE_THRESHOLD` (default: 0.3)
3. **Semantic**: No significant keyword overlap between query and top results

**Output Example:**

```python
{
    'is_insufficient': True,
    'reasons': [
        'Too few results (2 < 5)',
        'Low average score (0.25 < 0.30)'
    ],
    'metrics': {
        'result_count': 2,
        'avg_score': 0.25,
        'has_semantic_overlap': True
    }
}
```

#### Section 3: Query Execution Pipeline

**Responsibility**: Orchestrate the complete query processing flow

```python
def retrieve_documents(
    self,
    question: str,
    max_local_results: int = 5,
    use_web_search: bool = True,
    auto_reload_empty: bool = True
) -> Dict[str, Any]:
    """Document retrieval without RAG generation."""

def augment_response(
    self,
    question: str,
    documents: List[Dict]
) -> RAGResponse:
    """Generate RAG response from documents."""

def query(
    self,
    question: str,
    max_local_results: int = 5,
    use_web_search: bool = True,
    auto_reload_empty: bool = True
) -> RAGResponse:
    """Complete end-to-end query pipeline."""
```

**Query Execution Flow Diagram:**

```
query()
    ├─→ retrieve_documents()
    │       ├─→ check_database_health()
    │       ├─→ _search_locally()
    │       ├─→ detect_insufficiency_for_query()
    │       ├─→ (conditional) _search_web()
    │       └─→ _consolidate_documents()
    │
    └─→ augment_response()
            ├─→ RAGModule.generate()
            │       ├─→ PromptTemplateFactory.create()
            │       ├─→ OllamaProvider.generate()
            │       └─→ CitationExtractor.extract()
            └─→ Return RAGResponse
```

#### Section 4: Evaluation

**Responsibility**: System performance assessment on test sets

```python
def evaluate_test(
    self,
    test_spec: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate retrieval system with IR metrics."""
```

**Process:**
1. Load test queries (from parameter or `data/test_queries.json`)
2. For each test query:
   - Execute local search via `_search_locally()`
   - Extract document IDs from results
   - Compare against relevance judgments
3. Compute aggregate and per-query IR metrics

#### Section 5: Diagnostics and Logging

**Responsibility**: System introspection and operation tracing

```python
def get_status(self) -> Dict[str, Any]:
    """Complete system status and diagnostics."""

def _log_step(self, step: str, message: str) -> None:
    """Log significant orchestration step."""
```

---

## Component Interactions

### Component Dependency Graph

```
MainOrchestator (Orchestrator)
    ├── SRIPipeline (Local Search)
    │   ├── InvertedIndex (TF-IDF)
    │   ├── LSIModel (Semantic Search)
    │   ├── VectorStore (Embeddings)
    │   ├── RankingEngine (Result Ranking)
    │   └── Evaluator (Metrics)
    │
    ├── OllamaProvider (LLM Inference)
    ├── RAGModule (Answer Generation)
    │   ├── PromptTemplateFactory (Template Selection)
    │   ├── CitationExtractor (Citation Management)
    │   └── OutputParser (Response Formatting)
    │
    ├── SufficiencyChecker (Insufficiency Detection)
    ├── WebSearcher (Web Search)
    │   └── DuckDuckGo Backend
    │
    └── CrawlerCaller (Data Acquisition)
        ├── 6 Specialized Crawlers
        └── Document Consolidation
```

### Interaction Protocol: MainOrchestator → SRIPipeline

**Initialization:**
```python
pipeline = SRIPipeline(
    lsi_components=100,
    top_k=10,
    load_existing=True
)
```

**Indexing Documents:**
```python
# MainOrchestator.load_documents_from_crawlers()
documents = self.crawler_caller.consolidate_raw_to_documents()
self.pipeline.index(documents)  # ← Updates all internal indices
```

**Searching:**
```python
# MainOrchestator._search_locally()
results = self.pipeline.search(query, top_k=max_results)
# Returns: List[Dict] with keys: id, title, content, score, url
```

**Key Design**: SRIPipeline uses **multi-signal ranking** combining:
- LSI semantic similarity
- TF-IDF term frequency
- Vector similarity (embeddings)
- Field length normalization

### Interaction Protocol: MainOrchestator → RAGModule

**Initialization:**
```python
rag_module = RAGModule(
    llm=self.llm_provider,
    template_type="domain_specific"
)
```

**Document Preparation:**
```python
# MainOrchestator.augment_response()
# Ensures documents have standard fields
rag_documents = [
    {
        "id": doc.get("doc_id") or doc.get("id"),
        "title": doc.get("title", "Untitled"),
        "content": doc.get("content", ""),
        "url": doc.get("url"),
        "score": doc.get("score")
    }
    for doc in documents
]
```

**RAG Generation:**
```python
response = self.rag_module.generate(
    query=question,
    documents=rag_documents,
    temperature=0.7,
    max_tokens=1024
)
# Returns: RAGResponse(answer=str, citations=List[Citation])
```

### Interaction Protocol: MainOrchestator → Web Search

**Trigger Condition:**
```
if use_web_search and insufficiency['is_insufficient']:
    web_results = self._search_web(question)
```

**Web Search Execution:**
```python
results = self.web_searcher.search(query)[:max_results]
# Returns: List[Dict] with keys: id, title, content, url, score
```

**Persistence:**
```python
# After web search, persist results to documents.json
web_docs = [
    {
        "id": r.get('id'),
        "title": clean_scraped_text(r.get("title")),
        "content": clean_scraped_text(r.get("content")),
        "url": r.get("url"),
        "source": "web_augment",
        "date": datetime.now(timezone.utc).isoformat()
    }
    for r in web_results
]
self.crawler_caller.merge_documents(web_docs)
```

---

## Data Flow Pipelines

### Pipeline 1: Complete Query Processing

**Trigger**: `orchestrator.query("What is machine learning?")`

```
Input: Query String
  ↓
Step 1: Database Health Check
  ├─ Count indexed documents
  ├─ Check vector store availability
  └─ Trigger auto-reload if below threshold
  ↓
Step 2: Local Search
  ├─ Tokenize query
  ├─ Compute LSI similarity scores
  ├─ Compute TF-IDF scores
  ├─ Compute vector similarity scores
  ├─ Rank results (multi-signal)
  └─ Return top-k results
  ↓
Step 3: Insufficiency Detection
  ├─ Check result quantity (< 5?)
  ├─ Check result quality (avg score < 0.3?)
  ├─ Check semantic overlap
  └─ Emit: is_insufficient, reasons
  ↓
Step 4: Conditional Web Search
  if use_web_search AND is_insufficient:
    ├─ Search DuckDuckGo
    ├─ Clean scraped results
    ├─ Persist to documents.json
    └─ Add to document list
  ↓
Step 5: Document Consolidation
  ├─ Merge local + web results
  ├─ Deduplicate by content hash
  ├─ Preserve local priority (local before web)
  └─ Return consolidated list
  ↓
Step 6: RAG Generation
  ├─ Select prompt template
  ├─ Apply template with query + documents
  ├─ Invoke Ollama LLM
  ├─ Parse structured response
  ├─ Extract citations
  └─ Enrich with document metadata
  ↓
Output: RAGResponse(answer, citations, metadata)
```

### Pipeline 2: Document Loading and Indexing

**Trigger**: `orchestrator.load_documents_from_crawlers()`

```
Input: max_articles=1000, force_recrawl=False
  ↓
Step 1: Check Existing Consolidated Documents
  if documents.json exists AND !force_recrawl:
    └─ Load and skip crawlers
  ↓
Step 2: Execute Crawlers (if needed)
  ├─ DevTo Spider (tech articles)
  ├─ HackerNews Spider (tech news)
  ├─ RealPython Spider (Python resources)
  ├─ Lobsters Spider (software engineering)
  ├─ TheNewStack Spider (cloud-native)
  └─ TheVerge Spider (tech reviews)
  ↓
Step 3: Consolidate Raw Data
  ├─ Read data/raw/*.json files
  ├─ Deduplicate by content hash
  ├─ Standardize field names
  └─ Save to data/documents.json
  ↓
Step 4: Index Documents
  ├─ Build inverted index (vocabulary + posting lists)
  ├─ Compute TF-IDF statistics
  ├─ Fit LSI model (100 components)
  ├─ Compute document embeddings
  ├─ Store in VectorStore
  └─ Persist to disk
  ↓
Output: {
    success: bool,
    total_documents: int,
    indexed_documents: int,
    duration_seconds: float
}
```

### Pipeline 3: Evaluation

**Trigger**: `orchestrator.evaluate_test(test_spec)`

```
Input: Test Queries
  ↓
Step 1: Load Test Set
  ├─ Use provided test_spec OR
  └─ Load data/test_queries.json
  ↓
Step 2: For Each Test Query
  ├─ Execute: _search_locally(query, top_k=10)
  ├─ Extract document IDs from results
  ├─ Compare against relevance judgments
  ├─ Compute query-specific metrics (AP, RR, P@k, etc.)
  └─ Record per-query results
  ↓
Step 3: Aggregate Metrics
  ├─ Compute MAP (mean average precision)
  ├─ Compute MRR (mean reciprocal rank)
  ├─ Compute mean P@1, P@3, P@5, P@10
  ├─ Compute mean R@1, R@3, R@5, R@10
  ├─ Compute mean NDCG@1, NDCG@3, NDCG@5, NDCG@10
  └─ Store aggregate statistics
  ↓
Output: {
    aggregate: Dict[str, float],
    per_query: List[Dict],
    execution_time_seconds: float
}
```

---

## Database Architecture

### Document Storage Hierarchy

```
documents.json (Single Source of Truth)
    │
    ├─→ VectorStore (In-Memory Index + Persistence)
    │   ├─ ChromaDB (if installed)
    │   │   └─ chroma.sqlite3
    │   └─ Custom Backend (fallback)
    │       └─ .npy files (embeddings)
    │
    ├─→ LSI Model
    │   └─ Serialized sklearn model
    │
    └─→ InvertedIndex
        └─ Vocabulary + posting lists
```

### Document Schema

**Standard Document Format:**

```python
{
    "id": str,              # Unique identifier
    "title": str,           # Document title
    "content": str,         # Full text content
    "url": str,             # Source URL (optional)
    "date": str,            # ISO 8601 timestamp
    "source": str,          # Source identifier (web_augment, devto, etc.)
    "tags": List[str],      # Optional metadata tags
    "score": float,         # Relevance score (set during search)
    "doc_id": str           # Alternative ID field (for legacy compatibility)
}
```

### State Management

**VectorStore State:**

```python
class VectorStore:
    def __init__(self, collection_name, persist_dir, use_chromadb):
        self._ids = []                    # Document IDs
        self._embeddings = np.array([])   # Embedding vectors
        self._metadata = {}               # Document metadata
        self.collection = None            # ChromaDB collection (if available)
```

**State Lifecycle:**

1. **Empty State**: VectorStore initialized, no documents
2. **Indexed State**: `index(documents)` called → embeddings computed → stored
3. **Searched State**: `search(query)` executed → query embedding computed → similarity computed
4. **Persisted State**: `save()` writes to disk → `chroma.sqlite3` or `.npy` files

**Critical Operations:**

```python
# Count documents (for health checks)
count = vstore.count()  # Uses ChromaDB API or _ids length

# Clear database (atomic)
vstore.clear_all()  # Wipes all indices and persistence

# Add documents (batch operation)
vstore.add(documents)  # Computes embeddings, stores references
```

---

## Search Methodology

### Multi-Method Ranking Algorithm

The SRIPipeline combines three independent ranking signals:

#### 1. LSI Semantic Similarity

**Algorithm**: Latent Semantic Indexing via Singular Value Decomposition

```
1. Represent query as TF-IDF vector
2. Project onto LSI space (100 dimensions)
3. Compute cosine similarity to all document projections
4. Normalize scores to [0, 1]
```

**Strengths**: Discovers semantic relationships beyond exact term matches
**Weakness**: Computationally expensive for large vocabularies

**Implementation:**
```python
# src/retrieval/lsi_model.py
class LSIModel:
    def __init__(self, n_components=100):
        self.svd = TruncatedSVD(n_components=n_components)
    
    def fit(self, documents):
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(...)
        matrix = tfidf.fit_transform(texts)
        # Fit SVD
        self.svd.fit(matrix)
    
    def search(self, query):
        # Project query to LSI space
        query_vector = self.tfidf.transform([query])
        query_lsi = self.svd.transform(query_vector)
        # Compute similarities
        similarities = cosine_similarity(query_lsi, self.doc_lsi)
        return similarities
```

#### 2. TF-IDF Ranking

**Algorithm**: Term Frequency - Inverse Document Frequency

```
score(q, d) = Σ_{t ∈ q} tf(t, d) * idf(t)
```

**Strengths**: Fast, interpretable, penalizes common terms
**Weakness**: No semantic understanding, exact term matches only

**Implementation:**
```python
# src/indexing/indexer.py
class InvertedIndex:
    def __init__(self, use_stemming=True):
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self._tokenize,
            stop_words='english'
        )
    
    def search(self, query):
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = query_vector.dot(self.tfidf_matrix.T)
        return similarities.toarray().flatten()
```

#### 3. Vector Similarity (Embeddings)

**Algorithm**: Semantic text embeddings via Sentence Transformers

```
1. Encode query and documents using pretrained model
2. Compute cosine similarity in embedding space
3. Normalize scores to [0, 1]
```

**Strengths**: State-of-the-art semantic understanding
**Weakness**: Slower inference, requires GPU for speed

**Implementation:**
```python
# src/retrieval/vector_store.py
class VectorStore:
    def __init__(self, use_chromadb=False):
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def search(self, query_embedding, top_k=10):
        # Compute similarity to all stored embeddings
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )
        # Return top-k indices
        return np.argsort(similarities[0])[-top_k:][::-1]
```

#### Multi-Signal Combination

**Ranking Formula:**

```
final_score(q, d) = w_lsi * score_lsi(q, d) 
                  + w_tfidf * score_tfidf(q, d)
                  + w_vector * score_vector(q, d)

where w_lsi = 0.4, w_tfidf = 0.3, w_vector = 0.3
```

**Implementation:**
```python
# src/ranking/ranking.py
class RankingEngine:
    def combine_scores(self, lsi_scores, tfidf_scores, vector_scores):
        weights = {'lsi': 0.4, 'tfidf': 0.3, 'vector': 0.3}
        
        combined = (
            weights['lsi'] * lsi_scores +
            weights['tfidf'] * tfidf_scores +
            weights['vector'] * vector_scores
        )
        
        # Normalize to [0, 1]
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-10)
        
        return combined
```

---

## RAG Pipeline

### Architecture

```
Query + Documents
    ↓
[PromptTemplateFactory]
    ├─ basic: Simple template with document summaries
    ├─ domain_specific: Tech-focused context and instructions
    └─ chain_of_thought: Multi-step reasoning prompt
    ↓
[Ollama LLM]
    ├─ Temperature: 0.7 (balanced creativity and determinism)
    ├─ Max tokens: 1024
    ├─ Model: llama3.2:latest (customizable)
    └─ Inference: ~100-500ms per query
    ↓
[CitationExtractor]
    ├─ Parse structured output from LLM
    ├─ Extract document references
    ├─ Enrich with source metadata
    └─ Format citations
    ↓
[OutputParser]
    ├─ Validate response structure
    ├─ Type conversion (RAGResponse)
    └─ Fallback error handling
    ↓
RAGResponse (answer + citations)
```

### Prompt Templates

#### 1. Basic Template

```
You are a helpful information retrieval assistant.

Documents:
{documents_text}

Question: {query}

Answer the question based on the provided documents.
```

#### 2. Domain-Specific Template

```
You are an expert in technology and software engineering.

Retrieved Documents:
{numbered_documents}

User Question: {query}

Instructions:
- Provide a comprehensive answer based on the documents
- Cite sources for all claims
- If information is incomplete, acknowledge the limitation
- Focus on technical accuracy

Answer:
```

#### 3. Chain-of-Thought Template

```
You are a reasoning assistant for technology questions.

Documents:
{documents_with_indices}

Question: {query}

Please answer this step-by-step:
1. Identify the key aspects of the question
2. Review relevant information in the documents
3. Synthesize an answer with specific citations
4. Verify the answer is supported by the documents

Step-by-Step Answer:
```

### Citation Extraction

**Algorithm:**

```
1. Parse LLM output for citation markers ([1], [2], etc.)
2. For each citation marker:
   - Extract document index
   - Retrieve source metadata (title, URL)
   - Find supporting text snippet
3. Format as structured Citation objects
```

**Citation Object:**

```python
class Citation:
    def __init__(self, 
                 doc_id: str,
                 title: str,
                 url: str,
                 snippet: str):
        self.doc_id = doc_id
        self.title = title
        self.url = url
        self.snippet = snippet
        self.date = None  # Optional: from metadata
```

**Implementation:**

```python
# src/rag/citations.py
class CitationExtractor:
    def extract(self, response_text, documents):
        citations = []
        
        # Find all citation markers [N]
        pattern = r'\[(\d+)\]'
        matches = re.finditer(pattern, response_text)
        
        for match in matches:
            idx = int(match.group(1)) - 1  # 1-indexed in response
            
            if 0 <= idx < len(documents):
                doc = documents[idx]
                citation = Citation(
                    doc_id=doc.get('id'),
                    title=doc.get('title'),
                    url=doc.get('url'),
                    snippet=self._extract_snippet(doc)
                )
                citations.append(citation)
        
        return citations
```

---

## Error Handling and Resilience

### Error Categories

#### 1. Database Errors

**Scenario**: VectorStore unavailable

```python
try:
    count = self.pipeline.vstore.count()
except Exception:
    # Fallback: Check internal state
    count = len(self.pipeline.vstore._ids)
```

**Scenario**: Document loading fails

```python
load_result = self.load_documents_from_crawlers()
if not load_result['success']:
    # Log error but don't crash
    logger.error(f"Load failed: {load_result['message']}")
    # Return graceful error response
```

#### 2. Search Failures

**Scenario**: Query returns no results

```python
local_results = self._search_locally(question)
if not local_results:
    if use_web_search:
        # Fall back to web search
        web_results = self._search_web(question)
    else:
        # Return empty response
        return {'documents': [], 'error': 'No results found'}
```

**Scenario**: Search raises exception

```python
try:
    results = self.pipeline.search(query, top_k=max_results)
except Exception as e:
    logger.warning(f"Local search failed: {str(e)}")
    return []  # Empty fallback
```

#### 3. LLM Failures

**Scenario**: Ollama service unreachable

```python
try:
    rag_response = self.augment_response(question, documents)
except Exception as e:
    logger.error(f"RAG generation failed: {e}")
    # Return raw documents without RAG
    return RAGResponse(
        answer="Error generating response",
        citations=[],
        metadata={'error': str(e)}
    )
```

**Scenario**: LLM timeout

```python
response = self.llm_provider.generate(
    prompt=prompt,
    temperature=0.7,
    max_tokens=1024,
    timeout=30  # 30-second timeout
)
```

#### 4. Web Search Failures

**Scenario**: DuckDuckGo API failure

```python
try:
    results = self.web_searcher.search(query)
except Exception as e:
    logger.warning(f"Web search failed: {str(e)}")
    return []  # Continue with local results only
```

### Graceful Degradation Strategy

```
[User Query]
  ↓
[Try: Local Search]
  └─ Success? → [Check Insufficiency]
                 └─ Sufficient? → [Generate Answer]
                              ↓
                         Success? → [Return Response]
                             ↓
                         Failure? → [Return RAG Error]
                 └─ Insufficient? → [Try: Web Search]
                                    └─ Success? → [Consolidate]
                                               ↓
                                         [Generate Answer]
                                    └─ Failure? → [Skip Web]
                                               ↓
                                         [Use Local Only]
  └─ Failure? → [Return Search Error]
```

---

## Performance Considerations

### Complexity Analysis

**Local Search:**
- Indexing: O(n * m) where n = documents, m = avg tokens/doc
- LSI fitting: O(n * m + k³) where k = components (100)
- Search: O(k + n) per query

**RAG Generation:**
- Prompt assembly: O(d * t) where d = documents, t = avg tokens
- LLM inference: ~500ms-5s per query (highly variable)
- Citation extraction: O(c * d) where c = citations, d = documents

**Bottlenecks:**

1. **LLM Inference** (60-80% of query time)
   - Mitigation: Use smaller models, batching, caching

2. **Vector Similarity** (15-25% of query time)
   - Mitigation: Approximate nearest neighbor (HNSW in ChromaDB)

3. **Web Search** (10-20% of query time when triggered)
   - Mitigation: Optional disabling, async execution

### Memory Usage

**Typical System (1000 documents):**

- VectorStore embeddings: ~50MB (1000 docs × 384 dims × 4 bytes)
- LSI model: ~3MB (100 components × vocabulary size)
- InvertedIndex: ~2MB (vocabulary + posting lists)
- LLM (Ollama): ~4GB (llama3.2:latest quantized)

**Total**: ~4GB RAM recommended

### Optimization Techniques

#### 1. Document Batching

```python
# Index documents in batches to reduce memory peaks
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    self.pipeline.index(batch)
```

#### 2. Query Caching

```python
# Cache identical queries
cache = {}

def query_with_cache(question):
    if question in cache:
        return cache[question]
    
    response = orchestrator.query(question)
    cache[question] = response
    return response
```

#### 3. Lazy LSI Loading

```python
# Don't load LSI until first search
if not self.pipeline.lsi.is_fitted:
    # Load from disk or fit from consolidated documents
    pass
```

---

## Extension Points

### 1. Adding New LLM Providers

**Interface:**

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response."""
    
    @abstractmethod
    def get_metadata(self) -> Dict:
        """Return provider metadata."""
```

**Example: HuggingFace Integration**

```python
class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name="meta-llama/Llama-2-7b"):
        self.pipeline = pipeline("text-generation", model=model_name)
    
    def generate(self, prompt: str, **kwargs) -> str:
        outputs = self.pipeline(prompt, **kwargs)
        return outputs[0]['generated_text']
```

**Integration:**

```python
# In MainOrchestrator.__init__
# self.llm_provider = HuggingFaceProvider(model_name="...")
```

### 2. Adding New Search Methods

**Example: BM25 Ranking**

```python
from rank_bm25 import BM25Okapi

class BM25Model:
    def __init__(self):
        self.corpus = []
        self.bm25 = None
    
    def fit(self, documents):
        self.corpus = documents
        tokenized_corpus = [doc['content'].split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        return scores
```

**Integration into RankingEngine:**

```python
# Add fourth ranking signal
combined = (
    0.3 * lsi_scores +
    0.3 * tfidf_scores +
    0.2 * vector_scores +
    0.2 * bm25_scores
)
```

### 3. Custom Prompt Templates

```python
# src/rag/prompt_templates.py
class CustomTemplate(BasePromptTemplate):
    def apply(self, query: str, documents: List[Dict]) -> Tuple[str, Dict]:
        # Custom logic here
        prompt = f"""
        Your custom template here
        Query: {query}
        Documents: {documents}
        """
        return prompt, {}

# Register template
PromptTemplateFactory.register("custom", CustomTemplate)

# Use
rag = RAGModule(llm, template_type="custom")
```

### 4. Custom Crawlers

```python
# src/sri/crawler/spiders/custom.py
import scrapy

class CustomSpider(scrapy.Spider):
    name = "custom"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com"]
    
    def parse(self, response):
        for article in response.css('article'):
            yield {
                'title': article.css('h2::text').get(),
                'content': ' '.join(article.css('p::text').getall()),
                'url': response.url,
                'date': datetime.now().isoformat()
            }
```

---

**Document Version**: 1.0  
**Last Updated**: May 31, 2026  
**Author**: Backend Engineering Team
