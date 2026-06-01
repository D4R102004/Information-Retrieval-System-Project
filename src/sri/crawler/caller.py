"""
Crawler Coordinator - Manage Multiple Data Sources

This module orchestrates the execution of all web crawlers (spiders) in
the system, consolidates their output into a unified documents collection,
and provides progress tracking and error handling.

Responsibilities:
  1. Execute all spiders sequentially or in parallel
  2. Consolidate raw JSON files (data/raw/{source}/*.json) into documents
  3. Save consolidated output to data/documents.json
  4. Provide detailed reports on crawling progress and errors
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from sri.crawler.spiders.devto import DevToSpider
from sri.crawler.spiders.hackernews import HackerNewsSpider
from sri.crawler.spiders.lobsters import LobstersSpider
from sri.crawler.spiders.realpython import RealPythonSpider
from sri.crawler.spiders.thenewstack import TheNewStackSpider
from sri.crawler.spiders.theverge import TheVergeSpider
from sri.crawler.pipeline import JsonPipeline


logger = logging.getLogger(__name__)

EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]",
    flags=re.UNICODE,
)
FRONTMATTER_PATTERN = re.compile(r"---\s*\n.*?\n---\s*\n", flags=re.DOTALL)
METADATA_KEYS = {
    "title:", "published:", "description:", "tags:", "canonical_url:",
    "cover_image:", "slug:", "reading_time:", "author:", "date:",
}


def _strip_emojis(text: str) -> str:
    """Remove common emoji symbols from text."""
    return EMOJI_PATTERN.sub("", text)


def clean_scraped_text(text: str) -> str:
    """
    Remove markdown frontmatter, metadata lines, and emoji symbols.

    Handles content from various sources (DevTo, etc.) that may include:
    - YAML frontmatter blocks (---...---)
    - Metadata key-value lines (title:, description:, etc.)
    - Emoji symbols
    """
    if not text:
        return ""

    # Remove YAML frontmatter blocks
    cleaned = FRONTMATTER_PATTERN.sub("", text)

    # Remove metadata key-value lines (lines starting with known metadata keys)
    lines = cleaned.split("\n")
    filtered_lines = []
    for line in lines:
        # Skip lines that are metadata (start with key:)
        is_metadata = any(line.strip().lower().startswith(key) for key in METADATA_KEYS)
        if not is_metadata and line.strip():
            filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines)

    # Remove emoji symbols
    cleaned = _strip_emojis(cleaned)

    return cleaned.strip()


class CrawlerCaller:
    """
    Orchestrates execution of all project spiders and output consolidation.

    This class manages:
      - Spider instantiation with configuration
      - Sequential or parallel execution
      - Raw output consolidation (raw/ → documents.json)
      - Progress tracking and reporting
      - Error handling and recovery
    """

    SUPPORTED_SPIDERS = [
        ("DevTo", DevToSpider),
        ("HackerNews", HackerNewsSpider),
        ("RealPython", RealPythonSpider),
        ("Lobsters", LobstersSpider),
        ("TheNewStack", TheNewStackSpider),
        ("TheVerge", TheVergeSpider),
    ]

    def __init__(
        self,
        raw_dir: str = "data/raw",
        documents_output: str = "data/documents.json",
        initial_corpus_dir: str = "dara/initial-corpus"
    ):
        """
        Initialize crawler coordinator.

        Args:
            raw_dir: Directory where raw JSON files are stored
            documents_output: Path to consolidated documents file
        """
        self.raw_dir = Path(raw_dir)
        self.initial_corpus_dir = Path(initial_corpus_dir)
        self.documents_output = Path(documents_output)
        self.pipeline = JsonPipeline(output_directory=str(raw_dir))

    def run_all_crawlers(self, max_articles: int = 500) -> Dict[str, Any]:
        """
        Execute all spiders and collect articles.

        Each spider runs independently and saves results to data/raw/{source}/.
        Progress is tracked per spider.

        Args:
            max_articles: Maximum articles per spider

        Returns:
            Dictionary with execution report:
            {
                "status": "success|error",
                "total_articles": int,
                "per_spider": {spider_name: count},
                "errors": [str],
                "time_seconds": float,
                "timestamp": str
            }
        """
        start_time = time.time()
        execution_report = {
            "status": "success",
            "total_articles": 0,
            "per_spider": {},
            "errors": [],
            "time_seconds": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Starting crawler execution with max_articles={max_articles}")

        try:
            for spider_name, spider_class in self.SUPPORTED_SPIDERS:
                try:
                    logger.info(f"Running {spider_name}...")
                    spider = spider_class(max_articles=max_articles)
                    articles = spider.fetch_articles()

                    # Save articles via pipeline
                    for article in articles:
                        self.pipeline.save_item(article)

                    count = len(articles)
                    execution_report["per_spider"][spider_name] = count
                    execution_report["total_articles"] += count

                    logger.info(f"  {spider_name}: {count} articles saved")

                    # Save last crawled time
                    with open(f"{self.raw_dir}/{spider_name.lower()}/_metadata.txt", "a", encoding = "utf-8") as metadata:
                        metadata.write(f"{datetime.now().isoformat()} {count}")

                except Exception as e:
                    error_msg = f"{spider_name} failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    execution_report["errors"].append(error_msg)

        except Exception as e:
            error_msg = f"Crawler execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            execution_report["status"] = "error"
            execution_report["errors"].append(error_msg)

        execution_report["time_seconds"] = time.time() - start_time

        logger.info(
            f"Crawler execution completed: {execution_report['total_articles']} "
            f"articles in {execution_report['time_seconds']:.2f}s"
        )

        return execution_report

    def consolidate_raw_to_documents(self, use_initial_corpus: bool = True) -> List[Dict[str, Any]]:
        """
        Load and consolidate all raw JSON files into unified document list.

        Scans data/raw/{source}/*.json and merges into single collection,
        ensuring documents have required fields and valid structure.

        Args:
            use_initial_corpus: If True, consolidate documents from initial corpus directory

        Returns:
            List of consolidated document dictionaries
        """
        consolidated: Dict[str, Dict[str, Any]] = {}
        source_dirs = []

        logger.info(f"Consolidating raw documents from {self.raw_dir}{(" and from " + str(self.initial_corpus_dir) if use_initial_corpus else '')}...")

        # Include initial corpus if enabled and exists
        if use_initial_corpus and self.initial_corpus_dir and self.initial_corpus_dir.is_dir():
            source_dirs.append(self.initial_corpus_dir)
        
        # Add all source directories from raw_dir
        source_dirs.extend([source_dir for source_dir in self.raw_dir.glob("*/") if source_dir.is_dir()])
        
        # Iterate through source directories
        for source_dir in source_dirs:
            source_name = source_dir.name
            count = 0

            try:
                # Load each JSON file in source directory
                for json_file in source_dir.glob("*.json"):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            doc = json.load(f)

                        # Ensure required fields
                        if not all(k in doc for k in ["id", "title", "content"]):
                            logger.warning(
                                f"Skipping {json_file.name}: missing required fields"
                            )
                            continue

                        # Add source if missing
                        if "source" not in doc:
                            doc["source"] = source_name

                        # Normalize stored text so downstream RAG receives clean content
                        doc["title"] = clean_scraped_text(str(doc.get("title", "")))
                        doc["content"] = clean_scraped_text(str(doc.get("content", "")))

                        # Deduplicate by ID across sources, prefer first occurrence
                        if doc.get("id") and doc["id"] not in consolidated:
                            consolidated[doc["id"]] = doc
                            count += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {json_file}: {e}")
                        continue

                logger.info(f"Consolidated {count} documents from {source_name}")

            except Exception as e:
                logger.error(f"Error consolidating {source_name}: {e}")

        logger.info(f"Total consolidated: {len(consolidated)} documents")
        return list(consolidated.values())

    def save_consolidated_documents(
        self, documents: List[Dict[str, Any]]
    ) -> Path:
        """
        Save consolidated documents to documents.json.

        Args:
            documents: List of document dictionaries

        Returns:
            Path to saved file
        """
        self.documents_output.parent.mkdir(parents=True, exist_ok=True)

        with open(self.documents_output, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(documents)} documents to {self.documents_output}")
        return self.documents_output

    def clear_cached_documents(self) -> Dict[str, Any]:
        """
        Remove cached crawler artifacts and consolidated documents.

        Deletes all files under data/raw and removes data/documents.json so the
        next load must crawl the web again.

        Returns:
            Dictionary with deletion counts and status.
        """
        deleted_files = 0
        deleted_dirs = 0

        if self.raw_dir.exists():
            for path in sorted(self.raw_dir.rglob("*"), reverse=True):
                try:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                        deleted_files += 1
                    elif path.is_dir():
                        try:
                            path.rmdir()
                            deleted_dirs += 1
                        except OSError:
                            # Directory not empty yet; keep walking.
                            pass
                except FileNotFoundError:
                    continue

        if self.documents_output.exists():
            self.documents_output.unlink()
            deleted_files += 1

        # Recreate the raw root so future crawls can write into it.
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Cleared crawler cache: {deleted_files} files, {deleted_dirs} directories"
        )
        return {
            "status": "success",
            "deleted_files": deleted_files,
            "deleted_directories": deleted_dirs,
        }

    def load_consolidated_documents(self) -> List[Dict[str, Any]]:
        """
        Load documents from the consolidated documents.json file.

        Returns:
            List of document dictionaries, or an empty list if the file is
            missing, invalid, or does not contain a list.
        """
        if not self.documents_output.exists():
            return []

        try:
            with open(self.documents_output, "r", encoding="utf-8") as f:
                documents = json.load(f)
            return documents if isinstance(documents, list) else []
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load consolidated documents: {e}")
            return []

    def merge_documents(
        self, new_documents: List[Dict[str, Any]]
    ) -> Path:
        """
        Merge new documents into existing documents.json without overwriting.

        Loads existing documents, adds new ones, deduplicates by 'id' field,
        and saves back to documents.json. Preserves all existing documents
        unless explicitly superseded by duplicate ID.

        Args:
            new_documents: List of new document dictionaries to add

        Returns:
            Path to updated file
        """
        self.documents_output.parent.mkdir(parents=True, exist_ok=True)

        # Load existing documents
        existing = []
        if self.documents_output.exists():
            try:
                with open(self.documents_output, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existing = data if isinstance(data, list) else []
                logger.info(f"Loaded {len(existing)} existing documents for merge")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load existing documents: {e}. Starting fresh.")
                existing = []

        # Build dict by id for deduplication
        by_id = {doc.get("id"): doc for doc in existing}
        added_count = 0
        for doc in new_documents:
            doc_id = doc.get("id")
            if doc_id and doc_id not in by_id:
                by_id[doc_id] = doc
                added_count += 1
            elif doc_id in by_id:
                logger.debug(f"Document {doc_id} already exists; skipping duplicate")

        # Convert back to list and save
        merged = list(by_id.values())
        with open(self.documents_output, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Merged {added_count} new documents; total now {len(merged)} "
            f"in {self.documents_output}"
        )
        return self.documents_output

    def count_raw_documents(self, folder: str = "*") -> int:
        """
        Count unconsolidated documents in data/raw/.

        Args:
            folder: target folder name ("*" for every folder in raw_dir)

        Returns:
            Number of JSON files in raw directories
        """
        count = sum(1 for _ in self.raw_dir.glob(f"{folder}/*.json"))
        return count
    
    def count_initial_corpus_documents(self) -> int:
        """
        Count documents in the initial corpus directory.

        Returns:
            Number of JSON files in the initial corpus directory, or 0 if it doesn't exist.
        """
        if self.initial_corpus_dir and self.initial_corpus_dir.is_dir():
            count = sum(1 for _ in self.initial_corpus_dir.glob("*.json"))
            return count
        return 0
    
    def get_last_crawled(self, source: str) -> str:
        """
        Gets the date of the last crawl event for a given source.

        Args:
            source (str): target source name

        Returns:
            str: date of last crawl event in ISO format.
        """
        
        path = f"{self.raw_dir}/{source}/_metadata.txt"
        try:
            with open(path, "r", encoding="utf-8") as metadata:
                last_entry = metadata.readlines()[-1].strip()
            return last_entry.split(" ")[0]
        except (IOError, IndexError):
            logger.warning(f"Failed to read last crawl date for source: {source}")
            return ""

    def count_consolidated_documents(self) -> int:
        """
        Count documents in consolidated documents.json file.

        Returns:
            Number of documents or 0 if file doesn't exist
        """
        if not self.documents_output.exists():
            return 0

        try:
            with open(self.documents_output, "r", encoding="utf-8") as f:
                docs = json.load(f)
            return len(docs) if isinstance(docs, list) else 0
        except (json.JSONDecodeError, IOError):
            return 0

    def execute_full_pipeline(
        self, max_articles: int = 500, force_recrawl: bool = False, use_initial_corpus: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline: crawl → consolidate → save.

        Args:
            max_articles: Maximum articles per spider
            force_recrawl: If True, run crawlers even if raw data exists
            use_initial_corpus: If True, include initial corpus in the pipeline

        Returns:
            Execution report with full pipeline results
        """
        start_time = time.time()

        # Run crawlers if needed
        if force_recrawl or self.count_raw_documents() == 0:
            crawl_report = self.run_all_crawlers(max_articles=max_articles)
        else:
            crawl_report = {
                "status": "skipped",
                "total_articles": self.count_raw_documents(),
                "reason": "Raw data already exists",
            }

        # Consolidate and save
        documents = self.consolidate_raw_to_documents(use_initial_corpus)
        self.save_consolidated_documents(documents)

        total_time = time.time() - start_time

        report = {
            "status": "success",
            "crawl_report": crawl_report,
            "consolidated_documents": len(documents),
            "per_source": self._count_by_source(documents),
            "time_seconds": total_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Full pipeline completed: {len(documents)} documents in "
            f"{total_time:.2f}s"
        )

        return report

    @staticmethod
    def _count_by_source(documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count documents by source for reporting."""
        counts = {}
        for doc in documents:
            source = doc.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts
