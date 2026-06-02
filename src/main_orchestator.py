"""
Main Orchestrator - Complete End-to-End System Orchestration

Unified interface for the entire information retrieval and generation system:
  1. Local database search (SRIPipeline)
  2. Automatic database management (crawlers, indexing)
  3. Insufficiency detection (quantity, quality, semantic)
  4. Web search augmentation (conditional)
  5. RAG-based answer generation with citations

API is interface-agnostic: returns structured Dict/objects, no CLI dependencies.
Suitable for CLI, GUI, REST API, or other interfaces.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

from sri.pipeline import SRIPipeline
from sri.crawler.caller import CrawlerCaller, clean_scraped_text
from main_config import main_config
from rag.rag_module import RAGModule
from rag.llm_provider import OllamaProvider # Change to desired LLM provider
from rag.output_parser import RAGResponse
from rag.config import rag_config
from sri.web_search.checker import SufficiencyChecker
from sri.web_search.searcher import WebSearcher

logger = logging.getLogger(__name__)

class MainOrchestator:
    """
    Complete orchestration of the information retrieval and generation system.

    Manages the full pipeline from query input through RAG generation, handling:
    - Database state management (clearing, loading, health checks)
    - Intelligent insufficiency detection using multiple criteria
    - Automatic crawler execution when data is insufficient
    - Local and web search with smart fallback
    - Document consolidation and deduplication
    - RAG-based answer generation with citation tracking
    - Complete metadata tracking for auditability

    All methods return structured Dict/Pydantic objects for interface independence.
    """

    def __init__(self):
        """Initialize all system components."""
        self.pipeline = SRIPipeline()
        self.llm_provider = OllamaProvider()
        self.rag_module = RAGModule(llm=self.llm_provider)
        self.sufficiency_checker = SufficiencyChecker()
        self.web_searcher = WebSearcher()
        self.crawler_caller = CrawlerCaller()
        self.settings = main_config
        
        # Paths
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self.documents_path = self.data_dir / "documents.json"
        self.raw_data_dir = self.data_dir / "raw"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # ==================== DATABASE MANAGEMENT ====================

    def clear_all_indices(self, clear_raw: Optional[bool] = None) -> Dict[str, Any]:
        """
        Clear all indices, models, and database state.

        Atomically removes:
        - VectorStore (local backend)
        - ChromaDB collection (if available)
        - Cached LSI models
        - All indices

        Args:
            clear_raw: If True, also deletes raw JSON files from crawlers

        Returns:
            Dict with operation status: {'success': bool, 'message': str, 'timestamp': str}
        """
        self._log_step("clear_all_indices", "Starting complete database cleanup")

        if clear_raw is None:
            clear_raw = self.get_setting("clear_raw")
        
        try:
            if clear_raw:
                cache_result = self.crawler_caller.clear_cached_documents()
                self._log_step(
                    "clear_all_indices",
                    f"Cleared crawler cache: {cache_result.get('deleted_files', 0)} files"
                )

            self.pipeline.vstore.clear_all()
            self._log_step("clear_all_indices", "VectorStore cleared successfully")
            
            return {
                'success': True,
                'message': f"All indices, models{", raw data," if clear_raw else ""} and documents.json cleared successfully",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            error_msg = f"Failed to clear indices: {str(e)}"
            try:
                logger.error(error_msg)
            except NameError:
                import logging as log_module
                log_module.getLogger(__name__).error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def load_documents_from_crawlers(
        self,
        max_articles_per_spider: Optional[int] = None,
        force_recrawl: Optional[bool] = None,
        use_initial_corpus: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Execute full crawl→consolidate→index pipeline.

        Runs all 6 crawlers, consolidates raw JSON files into unified documents.json,
        and builds indices for local search.

        Args:
            max_articles: Maximum articles per spider
            force_recrawl: If True, ignore existing crawls and re-execute
            use_initial_corpus: If True, include documents from initial corpus directory

        Returns:
            Dict with execution results: {
                'success': bool,
                'total_documents': int,
                'indexed_documents': int,
                'duration_seconds': float,
                'message': str
            }
        """
        self._log_step("load_documents", "Starting crawler execution and indexing")
        start_time = time.time()

        if max_articles_per_spider is None:
            max_articles_per_spider = self.get_setting("max_articles_per_spider")
            logger.warning(f"max_articles_per_spider={max_articles_per_spider}")

        if force_recrawl is None:
            force_recrawl = self.get_setting("force_recrawl")
            logger.warning(f"force_recrawl={force_recrawl}")

        if use_initial_corpus is None:
            use_initial_corpus = self.get_setting("use_initial_corpus")

        try:
            documents = []

            if not force_recrawl:
                documents = self.crawler_caller.load_consolidated_documents()
                if documents:
                    self._log_step(
                        "load_documents",
                        f"Loaded {len(documents)} consolidated documents from documents.json"
                    )

            if not documents:
                # Execute crawlers only when we have no consolidated file
                crawl_result = self.crawler_caller.execute_full_pipeline(
                    force_recrawl=force_recrawl,
                    max_articles=max_articles_per_spider,
                    use_initial_corpus=use_initial_corpus
                )

                status = crawl_result.get('status') if isinstance(crawl_result, dict) else None
                if status not in ("success", "skipped"):
                    return {
                        'success': False,
                        'message': f"Crawler execution failed: {crawl_result}",
                        'total_documents': 0,
                        'indexed_documents': 0,
                        'duration_seconds': time.time() - start_time
                    }

                documents = self.crawler_caller.consolidate_raw_to_documents(use_initial_corpus=use_initial_corpus)
                self._log_step("load_documents", f"Consolidated {len(documents)} documents from raw data")

            if not documents:
                return {
                    'success': False,
                    'message': 'No documents available to index',
                    'total_documents': 0,
                    'indexed_documents': 0,
                    'duration_seconds': time.time() - start_time
                }

            # Index documents
            try:
                self.pipeline.index(documents)
                # Log VectorStore state after indexing to confirm persistence
                try:
                    vcount = self.pipeline.vstore.count()
                except Exception:
                    vcount = len(self.pipeline.vstore._ids) if hasattr(self.pipeline.vstore, '_ids') else 0

                self._log_step("load_documents", f"Indexed {len(documents)} documents; vstore.count()={vcount}")
            except Exception as e:
                logger.error(f"Indexing failed: {str(e)}")
                return {
                    'success': False,
                    'message': f'Indexing failed: {str(e)}',
                    'total_documents': len(documents),
                    'indexed_documents': 0,
                    'duration_seconds': time.time() - start_time
                }

            duration = time.time() - start_time
            return {
                'success': True,
                'total_documents': len(documents),
                'indexed_documents': len(documents),
                'duration_seconds': duration,
                'message': f'Successfully loaded and indexed {len(documents)} documents'
            }

        except Exception as e:
            error_msg = f"Crawler loading failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'total_documents': 0,
                'indexed_documents': 0,
                'duration_seconds': time.time() - start_time
            }

    def check_database_health(self) -> Dict[str, Any]:
        """
        Check database status and readiness.

        Returns:
            Dict with status: {
                'is_empty': bool,
                'document_count': int,
                'can_search': bool,
                'has_chromadb': bool,
                'status': str  # 'healthy', 'empty', 'degraded'
            }
        """
        try:
            # Count documents in VectorStore (use provided count() when available)
            try:
                vec_count = int(self.pipeline.vstore.count())
            except Exception:
                vec_count = len(self.pipeline.vstore._ids) if hasattr(
                    self.pipeline.vstore, '_ids'
                ) else 0

            # Count documents loaded in classic local indices (inverted index / LSI)
            try:
                index_count = int(getattr(self.pipeline.indexer, 'num_docs', 0) or 0)
            except Exception:
                index_count = 0

            effective_count = max(vec_count, index_count)

            # Check if documents file exists
            file_count = 0
            if self.documents_path.exists():
                try:
                    with open(self.documents_path, 'r', encoding='utf-8') as f:
                        docs = json.load(f)
                        file_count = len(docs) if isinstance(docs, list) else 0
                except Exception as e:
                    logger.debug(f"Failed to read documents file: {str(e)}")

            # Check ChromaDB
            has_chromadb = False
            try:
                has_chromadb = self.pipeline.vstore.chroma_available()
            except Exception as e:
                logger.debug(f"Failed to check ChromaDB: {str(e)}")

            is_empty = effective_count == 0 and file_count == 0
            can_search = effective_count >= self.settings["min_documents"]
            status = 'healthy' if can_search else ('empty' if is_empty else 'degraded')

            return {
                'is_empty': is_empty,
                'document_count': effective_count,
                'vector_document_count': vec_count,
                'indexed_document_count': index_count,
                'file_document_count': file_count,
                'can_search': can_search,
                'has_chromadb': has_chromadb,
                'status': status
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'is_empty': True,
                'document_count': 0,
                'file_document_count': 0,
                'can_search': False,
                'has_chromadb': False,
                'status': 'error'
            }

    def reindex_database(self, 
                         auto_reload: Optional[bool] = None, 
                         db_health: Optional[Dict[str, Any]] = None, 
                         use_initial_corpus: Optional[bool] = None
                         ) -> Dict[str, Any]:
        """
        Rebuild database indices to ensure minimum document threshold.

        Executes a cascading enrichment strategy:
        1. Load from consolidated documents.json if available
        2. Consolidate raw JSON files if present and necessary
        3. Execute full crawler pipeline if necessary

        Guarantees that after successful completion, the database contains
        at least self.settings["min_documents"] indexed documents.

        Args:
            use_crawlers: Whether to use crawlers for indexing
            db_health: Optional pre-fetched database health status to inform strategy
            use_initial_corpus: If True, include documents from initial corpus directory during consolidation

        Returns:
            Dict with reindexing status: {
                'success': bool,
                'indexed_documents': int,
                'duration_seconds': float,
                'crawled': bool,
                'message': str,
                'timestamp': str
            }
        """
        self._log_step("reindex_database", "Starting database reindex")
        start_time = time.time()

        if use_initial_corpus is None:
            use_initial_corpus = self.get_setting("use_initial_corpus")
        
        if auto_reload is None:
            auto_reload = self.get_setting("auto_reload")

        try:
            if db_health is None:
                db_health = self.check_database_health()
            doc_count = int(db_health.get('document_count', 0))
            file_count = int(db_health.get('file_document_count', 0))
            raw_count = self._count_raw_documents()
            initial_corpus_count = self._count_initial_corpus_documents()

            self._log_step(
                "reindex_database",
                f"DB state -> indexed:{doc_count}, consolidated:{file_count}, raw:{raw_count}, initial_corpus:{initial_corpus_count}"
            )

            initial_corpus_count = initial_corpus_count if use_initial_corpus else 0
            doc_count = 0
            crawled = False

            # Attempt 1: Load from consolidated documents.json
            if file_count >= self.settings["min_documents"]:
                self._log_step("reindex_database", f"Loading {file_count} documents from documents.json")
                try:
                    docs = self.crawler_caller.load_consolidated_documents()
                    if docs:
                        self.pipeline.index(docs)
                        doc_count = len(docs)
                        self._log_step("reindex_database", f"Indexed {doc_count} documents from consolidated file")
                except Exception as e:
                    logger.warning(f"Failed to index consolidated documents: {e}")

            # Attempt 2: Consolidate and index raw documents
            if doc_count < self.settings["min_documents"] and raw_count + initial_corpus_count >= self.settings["min_documents"]:
                self._log_step("reindex_database", f"Consolidating {raw_count} raw documents")
                try:
                    docs = self.crawler_caller.consolidate_raw_to_documents(use_initial_corpus=use_initial_corpus)
                    if docs:
                        self.crawler_caller.save_consolidated_documents(docs)
                        self.pipeline.index(docs)
                        doc_count = len(docs)
                        self._log_step("reindex_database", f"Consolidated and indexed {doc_count} raw documents")
                except Exception as e:
                    logger.warning(f"Failed to consolidate/index raw documents: {e}")

            # Attempt 3: Execute full crawler pipeline
            if doc_count < self.settings["min_documents"] and auto_reload:
                self._log_step("reindex_database", "Below minimum threshold; executing crawlers")
                try:
                    load_result = self.load_documents_from_crawlers(force_recrawl=True, use_initial_corpus=use_initial_corpus)
                    if load_result.get('success'):
                        doc_count = load_result.get('indexed_documents', 0)
                        self._log_step(
                            "reindex_database",
                            f"Crawlers loaded and indexed {doc_count} documents"
                        )
                        crawled = True
                    else:
                        self._log_step("reindex_database", f"Crawler pipeline failed: {load_result.get('message')}")
                except Exception as e:
                    logger.warning(f"Crawler execution failed: {e}")

            # Final health check
            success = doc_count >= self.settings["min_documents"]
            duration = time.time() - start_time

            message = (
                f"Database reindex completed with {doc_count} indexed documents"
                if success
                else f"Reindex incomplete: {doc_count} documents (required: {self.settings["min_documents"]})"
            )

            self._log_step("reindex_database", message)

            return {
                'success': success,
                'indexed_documents': doc_count,
                'duration_seconds': round(duration, 2),
                'crawled': crawled,
                'message': message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            error_msg = f"Database reindex failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'indexed_documents': 0,
                'duration_seconds': round(time.time() - start_time, 2),
                'crawled': False,
                'message': error_msg,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    # ==================== INSUFFICIENCY DETECTION ====================

    def _has_semantic_overlap(self, query: str, documents: List[Dict]) -> bool:
        """
        Detect semantic overlap between query keywords and documents.

        Args:
            query: User query text
            documents: List of document dicts with 'content' or 'text' fields

        Returns:
            True if significant keyword overlap detected
        """
        query_words = set(query.lower().split())
        query_words = {w for w in query_words if len(w) > 3}  # Filter short words

        if not query_words:
            return True  # Can't meaningfully check

        overlap_count = 0
        for doc in documents[:rag_config.max_cites]:  # Check first max_cites docs
            doc_text = self._extract_content(doc).lower()
            doc_words = set(doc_text.split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                overlap_count += 1

        return overlap_count > 0  # At least one doc has some overlap

    def _detect_insufficiency_for_query(
        self,
        query: str,
        results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Apply 3-criterion insufficiency detection.

        Criteria:
        1. Quantity: Too few results (< MIN_RESULTS_FOR_QUERY)
        2. Quality: Low average relevance score (< MIN_AVG_SCORE_THRESHOLD)
        3. Semantic Overlap: No significant keyword overlap with query

        Args:
            query: User query
            results: Local search results with relevance scores

        Returns:
            Dict with insufficiency assessment: {
                'is_insufficient': bool,
                'reasons': List[str],
                'metrics': {
                    'result_count': int,
                    'avg_score': float,
                    'has_semantic_overlap': bool
                }
            }
        """
        reasons = []
        metrics = {
            'result_count': len(results),
            'avg_score': 0.0,
            'has_semantic_overlap': False
        }

        # Criterion 1: Quantity
        if len(results) < self.settings["MIN_RESULTS_FOR_QUERY"]:
            reasons.append(
                f"Too few results ({len(results)} < {self.settings["MIN_RESULTS_FOR_QUERY"]})"
            )

        # Criterion 2: Quality
        if results:
            scores = [
                r.get('score', 0) for r in results
                if isinstance(r.get('score'), (int, float))
            ]
            if scores:
                avg_score = sum(scores) / len(scores)
                metrics['avg_score'] = avg_score
                if avg_score < self.settings["MIN_AVG_SCORE_THRESHOLD"]:
                    reasons.append(
                        f"Low average score ({avg_score:.2f} < {self.settings["MIN_AVG_SCORE_THRESHOLD"]})"
                    )

        # Criterion 3: Semantic overlap
        has_overlap = self._has_semantic_overlap(query, results)
        metrics['has_semantic_overlap'] = has_overlap
        if not has_overlap:
            reasons.append("Insufficient semantic overlap with query")

        return {
            'is_insufficient': len(reasons) > 0,
            'reasons': reasons,
            'metrics': metrics
        }

    # ==================== QUERY EXECUTION PIPELINE ====================

    def retrieve_documents(
        self,
        question: str,
        max_local_results: Optional[int] = None,
        max_web_results: Optional[int] = None,
        enable_web_search: Optional[bool] = None,
        auto_reload: Optional[bool] = None,
        use_initial_corpus: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a question.

        Executes: database readiness checks -> local search -> insufficiency detection ->
        conditional web search -> document consolidation.

        Args:
            question: User query
            max_local_results: Maximum local search results to use
            enable_web_search: Enable web search augmentation
            auto_reload: Auto-execute crawlers if DB is below minimum threshold
            use_initial_corpus: If True, include initial corpus documents in reindexing if necessary

        Returns:
            Dict with retrieval output: {
                'documents': List[Dict],
                'metadata': Dict[str, Any],
                'error': Optional[str]
            }
        """
        metadata = {
            'auto_crawled': False,
            'total_documents_used': 0,
            'local_documents': 0,
            'web_documents': 0,
            'insufficiency_detected': False,
            'insufficiency_reasons': [],
            'generation_time_seconds': 0.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        if not question or not question.strip():
            metadata['retrieved_documents'] = []
            return {
                'documents': [],
                'metadata': metadata,
                'error': "Error: Query cannot be empty"
            }

        self._log_step("retrieve_documents", f"Retrieving documents for: {question}")

        if max_local_results is None:
            max_local_results = self.get_setting("max_local_results")

        if max_web_results is None:
            max_web_results = self.get_setting("max_web_results")

        if enable_web_search is None:
            enable_web_search = self.get_setting("enable_web_search")

        if auto_reload is None:
            auto_reload = self.get_setting("auto_reload")

        if use_initial_corpus is None:
            use_initial_corpus = self.get_setting("use_initial_corpus")

        # Step 1: Check database health and ensure minimum documents before searching.
        db_health = self.check_database_health()
        doc_count = int(db_health.get('document_count', 0))
        file_count = int(db_health.get('file_document_count', 0))
        raw_count = self._count_raw_documents()
        initial_corpus_count = self._count_initial_corpus_documents()

        self._log_step(
            "retrieve_documents",
            f"DB counts -> indexed:{doc_count}, consolidated_file:{file_count}, raw:{raw_count}, initial_corpus:{initial_corpus_count}"
        )

        # Only allow search when there are at least self.settings["min_documents"] indexed documents.
        allowed_to_search = doc_count >= self.settings["min_documents"]

        # If not enough indexed docs, attempt reindexing strategies.
        if not allowed_to_search:
            reindex_result = self.reindex_database(auto_reload, db_health, use_initial_corpus)
            allowed_to_search = (reindex_result.get('success', False) and
                                reindex_result.get('indexed_documents', 0) >= self.settings["min_documents"]) # safety double-check
            metadata['auto_crawled'] = reindex_result.get('crawled', False)
            metadata['reindexing_details'] = reindex_result.get('message', '')

        db_health = self.check_database_health()

        # If still not allowed, return error if database is empty or proceed with warning if below quota but not empty
        if not allowed_to_search:
            doc_count = int(db_health.get('document_count', 0))
            self._log_step("retrieve_documents", f"Insufficient documents ({doc_count}) after reindexing attempts")

            if doc_count < 1:
                metadata['retrieved_documents'] = []
                return {
                    'documents': [],
                    'metadata': metadata,
                    'error': "Database is empty. A search is impossible to perform"
                }

        # Step 2: Local search
        local_results = self._search_locally(question, max_results=max_local_results)
        for result in local_results:
            if "content" not in result and "snippet" in result:
                result["content"] = result.pop("snippet") # Legacy support for snippet field
        self._log_step("retrieve_documents", f"Local search returned {len(local_results)} results")

        # Step 3: Insufficiency detection
        insufficiency = self._detect_insufficiency_for_query(question, local_results)
        insufficient = insufficiency['is_insufficient']
        metadata['insufficiency_detected'] = insufficient
        metadata['insufficiency_reasons'] = insufficiency['reasons']
        metadata['local_documents'] = len(local_results)

        # Step 4: Web search if needed
        web_results = []
        if enable_web_search and insufficient:
            self._log_step("retrieve_documents", "Insufficiency detected, performing web search")
            web_results = self._search_web(question, max_web_results)
            metadata['web_documents'] = len(web_results)
            self._log_step("retrieve_documents", f"Web search returned {len(web_results)} results")

        if web_results:
            # Persist web results to documents.json for future sessions
            web_docs = [
                {
                    "id": r.get('id', ''),
                    "title": clean_scraped_text(str(r.get("title", ""))),
                    "content": clean_scraped_text(str(r.get("content", ""))),
                    "url": r.get("url"),
                    "source": "web",
                    "date": datetime.now(timezone.utc).isoformat(),
                }
                for i, r in enumerate(web_results)
            ]
            try:
                self.crawler_caller.merge_documents(web_docs)
                self._log_step(
                    "retrieve_documents",
                    f"Merged {len(web_docs)} web results to documents.json "
                )
            except Exception as e:
                logger.warning(f"Failed to persist web results: {e}")
            
            # Persist web results for current session use without full reindex.
            fails = 0
            for doc in web_docs:
                try:
                    self.pipeline.add_document(doc)
                except Exception as e:
                    fails += 1
                    logger.warning(f"Failed to add web document {doc.get('id', 'unknown')} to active indices: {e}")

            logger.info(
                f"Added {len(web_docs) - fails} web documents to active indices; {fails} failures"
            )

        # Step 5: Consolidate documents
        all_documents = self._consolidate_documents(local_results, web_results)
        metadata['total_documents_used'] = len(all_documents)
        metadata['retrieved_documents'] = all_documents
        self._log_step("retrieve_documents", f"Consolidated {len(all_documents)} documents")

        return {
            'documents': all_documents,
            'metadata': metadata,
            'error': None
        }

    def augment_response(self, question: str, documents: List[Dict]) -> RAGResponse:
        """
        Generate a RAG response given a list of documents.

        Args:
            question: User query text used for prompt generation
            documents: List of consolidated documents to pass to RAG

        Returns:
            RAGResponse containing `answer`, `citations`, and minimal metadata
        """
        self._log_step("augment_response", f"Generating RAG answer for {len(documents)} documents")
        start = time.time()

        try:
            if not documents:
                return RAGResponse(answer="No relevant documents found to generate an answer.", citations=[])

            rag_resp = self.rag_module.generate(query=question, documents=documents)
            duration = time.time() - start
            self._log_step("augment_response", f"RAG generation completed in {duration:.2f}s")

            return rag_resp

        except Exception as e:
            logger.error(f"augment_response failed: {e}", exc_info=True)
            return RAGResponse(answer=f"Error generating response: {str(e)}", citations=[])

    def query(
        self,
        question: str,
        max_local_results: Optional[int] = None,
        max_web_results: Optional[int] = None,
        enable_web_search: Optional[bool] = None,
        auto_reload: Optional[bool] = None,
        use_initial_corpus: Optional[bool] = None
    ) -> RAGResponse:
        """
        Complete query execution pipeline.

        Executes: local search → insufficiency detection → conditional web search →
        document consolidation → RAG generation → citation extraction.

        Automatically loads data from crawlers if database is empty and
        auto_reload=True.

        Args:
            question: User query
            max_local_results: Maximum local search results to use
            enable_web_search: Enable web search augmentation
            auto_reload: Auto-execute crawlers if DB empty

        Returns:
            RAGResponse with answer, citations, and metadata.
            Retrieved documents available in response.metadata['retrieved_documents']
        """
        execution_start = time.time()
        metadata: Dict[str, Any] = {
            'generation_time_seconds': 0.0,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'retrieved_documents': []
        }

        if max_local_results is None:
            max_local_results = self.get_setting("max_local_results")

        if max_web_results is None:
            max_web_results = self.get_setting("max_web_results")

        if enable_web_search is None:
            enable_web_search = self.get_setting("enable_web_search")

        if auto_reload is None:
            auto_reload = self.get_setting("auto_reload")

        if use_initial_corpus is None:
            use_initial_corpus = self.get_setting("use_initial_corpus")

        try:
            self._log_step("query", f"Processing: {question}")

            retrieval_result = self.retrieve_documents(
                question=question,
                max_local_results=max_local_results,
                max_web_results=max_web_results,
                enable_web_search=enable_web_search,
                auto_reload=auto_reload,
                use_initial_corpus=use_initial_corpus
            )
            metadata = retrieval_result.get('metadata', metadata)
            all_documents = retrieval_result.get('documents', [])

            retrieval_error = retrieval_result.get('error')
            if retrieval_error:
                metadata['generation_time_seconds'] = time.time() - execution_start
                metadata.setdefault('retrieved_documents', [])
                return RAGResponse(
                    answer=retrieval_error,
                    citations=[],
                    metadata=metadata
                )

            self._log_step("query", f"Consolidated {len(all_documents)} documents for RAG")

            # RAG generation
            rag_begin = time.time()
            rag_response = self.augment_response(question=question, documents=all_documents)

            metadata['generation_time_seconds'] = time.time() - rag_begin
            metadata['retrieved_documents'] = all_documents

            rag_response = RAGResponse(
                answer=rag_response.answer,
                citations=rag_response.citations,
                metadata=metadata
            )

            self._log_step("query", f"Query completed in {time.time() - execution_start:.2f}s")
            return rag_response

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}", exc_info=True)
            metadata['generation_time_seconds'] = time.time() - execution_start
            metadata['retrieved_documents'] = []
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                citations=[],
                metadata=metadata
            )

    def _search_locally(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute local search via SRIPipeline.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of ranked document results with scores
        """
        if max_results is None:
            max_results = self.get_setting("max_local_results")        

        try:
            results = self.pipeline.search(query, top_k=max_results)
            return results if results else []
        except Exception as e:
            logger.warning(f"Local search failed: {str(e)}")
            return []

    def _search_web(
        self,
        query: str,
        max_results: int= 10
    ) -> List[Dict[str, Any]]:
        """
        Execute web search via DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of web search results
        """
        try:
            results = self.web_searcher.search(query)[:max_results]
            return results if results else []
        except Exception as e:
            logger.warning(f"Web search failed: {str(e)}")
            return []

    def _extract_content(self, document: Dict) -> str:
        """
        Extract content from document supporting multiple field names.

        Args:
            document: Document dict with potential content fields

        Returns:
            Content string from document
        """
        return (
            document.get('content') or
            document.get('text') or
            document.get('body') or
            ''
        )

    def _consolidate_documents(
        self,
        local_results: List[Dict],
        web_results: List[Dict]
    ) -> List[Dict]:
        """
        Consolidate and deduplicate local and web results.

        Args:
            local_results: Results from local search
            web_results: Results from web search

        Returns:
            Consolidated document list, prioritizing local over web results
        """
        consolidated = []
        seen_content = set() # for deduplication based on content hash

        # Add web results (priority as assumed to be fresher content)
        for result in web_results:
            content = self._extract_content(result)
            content_hash = hash(content[:100])
            if content_hash not in seen_content:
                consolidated.append(result)
                seen_content.add(content_hash)

        # Add local results (assumed sorted by score)
        for result in local_results:
            content = self._extract_content(result)
            content_hash = hash(content[:100])
            if content_hash not in seen_content:
                consolidated.append(result)
                seen_content.add(content_hash)

        return consolidated
    
    def _count_raw_documents(self) -> int:
        """
        Count total raw documents available across all crawler output directories.

        Returns:
            Total count of raw documents available for consolidation and indexing.
        """
        try:
            return self.crawler_caller.count_raw_documents()
        except Exception as e:
            logger.warning(f"Failed to count raw documents: {str(e)}")
            return 0
        
    def _count_initial_corpus_documents(self) -> int:
        """
        Count total documents available in the initial corpus directory.

        Returns:
            Total count of initial corpus documents available for consolidation and indexing.
        """
        try:
            return self.crawler_caller.count_initial_corpus_documents()
        except Exception as e:
            logger.warning(f"Failed to count initial corpus documents: {str(e)}")
            return 0
        
    def _count_consolidated_documents(self) -> int:
        """
        Count total documents available in the consolidated documents.json file.

        Returns:
            Total count of consolidated documents available for indexing.
        """
        try:
            return self.crawler_caller.count_consolidated_documents()
        except Exception as e:
            logger.warning(f"Failed to count consolidated documents: {str(e)}")
            return 0
        
    def _get_default(self, key: str) -> Any:
        return self.settings.default(key)
    
    def get_setting(self, key: str) -> Any:
        try:
            return self.settings[key]
        except Exception as e:
            logger.warning(str(e))
            default = self._get_default(key)
            logger.warning(f"Using default value: {key} = {default}")
            return default
            
    def sync_backend(self, state: dict[str, Any]):
        """Attempts to sync settings with backend configuration

        Args:
            state (dict[str, Any]): current state

        Returns:
            bool: Syncronization performed correctly
        """
        try:
            for key, val in state.items():
                self.settings[key] = val
            self._log_step("sync_backend", "Backend syncronization completed succesfully")
        except Exception as e:
            logger.error("Backend syncronization failed:", str(e))

    # ==================== EVALUATION ====================

    def evaluate_test(
        self,
        test_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the retrieval system on a test set with relevance judgments.

        Accepts an optional test specification dict. If None or incomplete,
        loads from data/test_queries.json. This approach allows both:
        - Frontend-provided test cases (ad-hoc evaluation)
        - Stored test cases (persistent evaluation benchmarks)

        Test specification format:
        {
            "test_queries": [
                {
                    "query_id": "q1",
                    "query": "search query text",
                    "relevant": ["doc_id1", "doc_id2"],
                    "grades": {"doc_id1": 3, "doc_id2": 2}  # optional: 0-3 scale
                }
            ]
        }

        Args:
            test_spec: Optional Dict with test configuration and test_queries.
                      If None or missing required fields, loads from
                      data/test_queries.json. Default: None

        Returns:
            Dict with evaluation results:
            {
                "aggregate": {
                    "num_queries": int,
                    "MAP": float,
                    "MRR": float,
                    "mean_P@1": float,
                    "mean_R@5": float,
                    "mean_NDCG@5": float,
                    ...
                },
                "per_query": [
                    {
                        "query_id": str,
                        "num_relevant": int,
                        "ap": float,
                        "rr": float,
                        "p@k": float,
                        ...
                    }
                ],
                "timestamp": str,
                "status": "success" | "error",
                "message": str,
                "execution_time_seconds": float
            }

        Raises:
            No exceptions raised; all errors captured in return status
        """
        from evaluation.evaluation import Evaluator

        try:
            self._log_step("evaluate_test", "Starting evaluation")

            # Step 1: Load or validate test queries
            test_queries = None

            if test_spec and isinstance(test_spec, dict):
                test_queries = test_spec.get("test_queries")

            # Step 2: Fallback to JSON file if not provided
            if not test_queries:
                test_queries_path = Path("data/test_queries.json")

                if test_queries_path.exists():
                    try:
                        with open(test_queries_path, "r", encoding="utf-8") as f:
                            stored_spec = json.load(f)
                            test_queries = stored_spec.get("test_queries", [])
                        self._log_step(
                            "evaluate_test",
                            f"Loaded {len(test_queries)} test queries from file"
                        )
                    except (IOError, json.JSONDecodeError) as e:
                        return {
                            "aggregate": {},
                            "per_query": [],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "status": "error",
                            "message": f"Failed to read test_queries.json: {str(e)}",
                            "execution_time_seconds": 0.0
                        }
                else:
                    return {
                        "aggregate": {},
                        "per_query": [],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "error",
                        "message": "No test specification provided and data/test_queries.json not found",
                        "execution_time_seconds": 0.0
                    }

            # Step 3: Validate test queries
            if not test_queries or not isinstance(test_queries, list):
                return {
                    "aggregate": {},
                    "per_query": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "message": "Test queries must be a non-empty list",
                    "execution_time_seconds": 0.0
                }

            # Step 4: Initialize evaluator and define retrieval function
            evaluator = Evaluator(k_values=[1, 3, 5, 10])

            def retrieval_function(query_text: str) -> List[str]:
                """
                Retrieve document IDs matching query using local search.

                Args:
                    query_text: Search query string

                Returns:
                    Ordered list of document IDs
                """
                results = self._search_locally(query_text, max_results=50)

                doc_ids = []
                for doc in results:
                    doc_id = doc.get("doc_id") or doc.get("id") or doc.get("source")
                    if doc_id:
                        doc_ids.append(str(doc_id))

                print("\n[EVALUATION DEBUG]")
                print("Query:", query_text)
                print("Retrieved IDs:", doc_ids)
                print("Retrieved titles:", [doc.get("title", "") for doc in results])

                return doc_ids

            # Step 5: Execute evaluation
            evaluation_start = time.time()

            aggregate = evaluator.evaluate_all(test_queries, retrieval_function)
            per_query_results = evaluator.results
            evaluation_time = time.time() - evaluation_start

            self._log_step(
                "evaluate_test",
                f"Evaluation completed: {len(test_queries)} queries in {evaluation_time:.2f}s"
            )

            return {
                "aggregate": aggregate,
                "per_query": per_query_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "message": f"Evaluated {len(test_queries)} queries successfully",
                "execution_time_seconds": round(evaluation_time, 3)
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            self._log_step("evaluate_test", f"Evaluation error: {str(e)}")

            return {
                "aggregate": {},
                "per_query": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "message": f"Evaluation error: {str(e)}",
                "execution_time_seconds": 0.0
            }

    # ==================== DIAGNOSTICS & LOGGING ====================

    def get_status(self) -> Dict[str, Any]:
        """
        Get complete system status and diagnostics.

        Returns:
            Dict with comprehensive system state
        """
        db_health = self.check_database_health()
        raw_count = self._count_raw_documents()
        initial_corpus_count = self._count_initial_corpus_documents()
        consolidated_count = self._count_consolidated_documents()

        return {
            'database': {
                'status': db_health['status'],
                'indexed_documents': db_health['document_count'],
                'file_documents': db_health['file_document_count'],
                'has_chromadb': db_health['has_chromadb'],
                'is_empty': db_health['is_empty']
            },
            'crawlers': {
                'initial_corpus_documents': initial_corpus_count,
                'raw_documents': raw_count,
                'consolidated_documents': consolidated_count
            },
            'pipeline': {
                'initialized': self.pipeline is not None,
                'has_lsi': hasattr(self.pipeline, 'lsi') and self.pipeline.lsi is not None
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _log_step(self, step: str, message: str) -> None:
        """Log a significant orchestration step."""
        logger.info(f"[{step}] {message}")
