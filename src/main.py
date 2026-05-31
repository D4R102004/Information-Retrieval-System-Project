"""
Integrated SRI-RAG System - Command-Line Interface

Complete end-to-end workflow CLI for the information retrieval and generation system:

  1. Local database search (LSI, vector similarity, TF-IDF)
  2. Sufficiency detection (quantity, quality, semantic criteria)
  3. Conditional web search augmentation (DuckDuckGo fallback)
  4. Document consolidation and deduplication
  5. RAG-based answer generation via Ollama
  6. Citation extraction and formatting

The system orchestrates all components automatically through MainOrchestrator,
providing a unified interface for complex question answering.

Usage:
    # Single query
    python main.py --query "How does machine learning work?"

    # Interactive mode
    python main.py --interactive

    # With custom parameters
    python main.py --query "Explain RAG" --max-local 5 --no-web-search

    # Database management
    python main.py --status
    python main.py --load-data
    python main.py --clear-db

Environment Requirements:
    - Ollama service running (default: http://localhost:11434)
    - Python 3.10+
"""

import sys
import os
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
from datetime import datetime

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(__file__))

# Import orchestrator and response model
from main_orchestator import MainOrchestator 
from rag.output_parser import RAGResponse

# =========================================================================
# Logging Configuration
# =========================================================================

def configure_logging(verbose: bool, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        verbose: Enable debug-level logging
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )
    
    return logging.getLogger(__name__)


# =========================================================================
# Database Operations
# =========================================================================

def display_database_status(orchestrator: MainOrchestator, logger: logging.Logger) -> None:
    """
    Display current database health and status.

    Args:
        orchestrator: MainOrchestator instance
        logger: Logger instance
    """
    health = orchestrator.check_database_health()
    
    print("\n" + "=" * 70)
    print("DATABASE STATUS")
    print("=" * 70)
    print(f"Status:              {health['status'].upper()}")
    print(f"Document Count:      {health['document_count']}")
    print(f"File Documents:      {health['file_document_count']}")
    print(f"Can Search:          {'Yes' if health['can_search'] else 'No'}")
    print(f"ChromaDB Available:  {'Yes' if health['has_chromadb'] else 'No'}")
    print("=" * 70 + "\n")
    
    logger.info(f"Database status: {health['status']}")


def load_data_command(orchestrator: MainOrchestator, logger: logging.Logger, max_articles: int = 100) -> None:
    """
    Load documents from crawlers and build indices.

    Args:
        orchestrator: MainOrchestator instance
        logger: Logger instance
        max_articles: Maximum articles per crawler
    """
    print("\n" + "=" * 70)
    print("LOADING DATA FROM CRAWLERS")
    print("=" * 70)
    
    logger.info(f"Starting crawler execution (max_articles={max_articles})")
    result = orchestrator.load_documents_from_crawlers(max_articles=max_articles)
    
    if result['success']:
        print(f"[OK] Success: {result['message']}")
        print(f"  Total documents:   {result['total_documents']}")
        print(f"  Indexed documents: {result['indexed_documents']}")
        print(f"  Duration:          {result['duration_seconds']:.2f}s")
        logger.info(f"Data loaded: {result['indexed_documents']} documents indexed")
    else:
        print(f"[FAIL] Failed: {result['message']}")
        logger.error(f"Data loading failed: {result['message']}")
    
    print("=" * 70 + "\n")


def clear_database_command(orchestrator: MainOrchestator, logger: logging.Logger) -> None:
    """
    Clear all indices, raw crawler files, and consolidated documents.

    Args:
        orchestrator: MainOrchestator instance
        logger: Logger instance
    """
    print("\n" + "=" * 70)
    print("CLEARING DATABASE")
    print("=" * 70)
    
    confirm = input(
        "WARNING: This will delete all indices, data/raw, and data/documents.json. Continue? (yes/no): "
    ).strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return
    
    logger.warning("User requested database clear")
    result = orchestrator.clear_all_indices()
    
    if result['success']:
        print(f"[OK] Success: {result['message']}")
        logger.info("Database cleared successfully")
    else:
        print(f"[FAIL] Failed: {result['message']}")
        logger.error(f"Database clear failed: {result['message']}")
    
    print("=" * 70 + "\n")


# =========================================================================
# Query Execution
# =========================================================================

def format_response(response: RAGResponse) -> None:
    """
    Format and display RAG response with citations.

    Args:
        response: RAGResponse object from MainOrchestrator
    """
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    citation_map = []
    for index, citation in enumerate(response.citations, 1):
        citation_id = getattr(citation, 'doc_id', None) or getattr(citation, 'id', None)
        if citation_id:
            citation_map.append((str(citation_id), index))

    shown_answer = response.answer
    for citation_id, index in sorted(citation_map, key=lambda item: len(item[0]), reverse=True):
        shown_answer = re.sub(rf"\[{re.escape(citation_id)}\]", f"[{index}]", shown_answer)

    print(shown_answer)
    
    # Display citations
    print("\n--- CITATIONS ---")
    if not response.citations:
        print("(No citations extracted)")
    else:
        for idx, citation in enumerate(response.citations, 1):
            title = getattr(citation, 'title', 'Untitled')
            url = getattr(citation, 'url', None)
            date = getattr(citation, 'date', None)
            snippet = getattr(citation, 'snippet', None)

            metadata_parts = []
            if date:
                metadata_parts.append(f"{date}")
            if url is not None:
                metadata_parts.append(f"{url}")

            print(f"[{idx}] {title} ({', '.join(metadata_parts)})")
            if snippet:
                print(f"    Snippet: {snippet}")

    # Display metadata if available
    if hasattr(response, 'metadata') and response.metadata:
        print("\n--- METADATA ---")
        metadata = response.metadata
        if isinstance(metadata, dict):
            print(f"Local documents used:  {metadata.get('local_documents', 0)}")
            print(f"Web documents used:    {metadata.get('web_documents', 0)}")
            print(f"Total documents:       {metadata.get('total_documents_used', 0)}")
            
            if metadata.get('insufficiency_detected'):
                print(f"Insufficiency:         Yes")
                print(f"Reasons:               {', '.join(metadata.get('insufficiency_reasons', []))}")
            
            if metadata.get('generation_time_seconds'):
                print(f"Generation time:       {metadata['generation_time_seconds']:.2f}s")
    
    print("=" * 70 + "\n")


def execute_single_query(
    orchestrator: MainOrchestator,
    query: str,
    max_local: int = 5,
    use_web_search: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Execute a single query through the orchestrator.

    Args:
        orchestrator: MainOrchestator instance
        query: User question
        max_local: Maximum local search results
        use_web_search: Enable web search fallback
        logger: Logger instance
    """
    if not logger:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Query received: {query[:60]}...")
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        
        response = orchestrator.query(
            question=query,
            max_local_results=max_local,
            use_web_search=use_web_search,
            auto_reload_empty=True,
        )
        
        format_response(response)
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        print(f"\n[ERROR] {e}\n")
def interactive_mode(orchestrator: MainOrchestator, logger: logging.Logger) -> None:
    """
    Run interactive query loop.

    Args:
        orchestrator: MainOrchestator instance
        logger: Logger instance
    """
    logger.info("Entering interactive mode")
    
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Information Retrieval System")
    print("=" * 70)
    print("Commands:")
    print("  ask <query>      - Ask a question")
    print("  status           - Show database status")
    print("  load             - Load data from crawlers")
    print("  clear            - Clear database")
    print("  help             - Show this help")
    print("  exit             - Exit application")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = input(">> ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd in {"exit", "quit", "bye", "q"}:
                print("Goodbye!")
                logger.info("User exited application")
                break
            
            elif cmd in {"status", "info"}:
                display_database_status(orchestrator, logger)
            
            elif cmd == "load":
                load_data_command(orchestrator, logger)
            
            elif cmd == "clear":
                clear_database_command(orchestrator, logger)
            
            elif cmd in {"ask", "query", "q"}:
                if not args:
                    print("Usage: ask <question>")
                    continue
                execute_single_query(orchestrator, args, logger=logger)
            
            elif cmd == "help":
                print("""
Available Commands:
  ask <query>      - Ask a question to the system
  status           - Display current database status
  load             - Load documents from crawlers
  clear            - Clear all indices (requires confirmation)
  help             - Show this help message
  exit             - Exit the application
                """)
            
            else:
                # Treat input as query if not a recognized command
                if len(user_input) > 3:
                    execute_single_query(orchestrator, user_input, logger=logger)
                else:
                    print("Command not recognized. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n(Interrupted)")
            logger.info("Interactive session interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"Error: {e}\n")


# =========================================================================
# CLI Configuration
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="SRI-RAG System: Integrated Information Retrieval & Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Query:        python main.py --query "How does RAG work?"
  Interactive:  python main.py --interactive
  Status:       python main.py --status
  Load data:    python main.py --load-data
  Clear DB:     python main.py --clear-db --force
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--query",
        type=str,
        help="Execute single query and exit",
        metavar="QUESTION",
    )
    mode_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive query mode",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Display database status",
    )
    mode_group.add_argument(
        "--load-data",
        action="store_true",
        help="Load documents from crawlers",
    )
    mode_group.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear all indices (requires --force)",
    )

    # Search options
    search_group = parser.add_argument_group("Search Options")
    search_group.add_argument(
        "--max-local",
        type=int,
        default=5,
        help="Maximum local search results to use (default: 5)",
        metavar="N",
    )
    search_group.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search fallback",
    )

    # Data options
    data_group = parser.add_argument_group("Data & Logging")
    data_group.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="Maximum articles per crawler (default: 100)",
        metavar="N",
    )
    data_group.add_argument(
        "--log-file",
        help="Write logs to file",
        metavar="PATH",
    )
    data_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    data_group.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmations (e.g., for --clear-db)",
    )

    return parser


# =========================================================================
# Main Application
# =========================================================================

def main() -> int:
    """
    Main application entry point.

    Parses arguments and delegates to appropriate operation:
    - Single query execution
    - Interactive mode
    - Database status display
    - Data loading
    - Database clearing

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    logger = configure_logging(args.verbose, log_file=args.log_file)

    try:
        # Initialize MainOrchestrator
        logger.info("Initializing MainOrchestrator...")
        orchestrator = MainOrchestator()  # type: ignore
        logger.info("MainOrchestrator initialized successfully")
        print("[OK] System ready\n")

        # Delegate to appropriate mode
        if args.query:
            execute_single_query(
                orchestrator,
                args.query,
                max_local=args.max_local,
                use_web_search=not args.no_web_search,
                logger=logger,
            )
            return 0

        elif args.interactive:
            interactive_mode(orchestrator, logger)
            return 0

        elif args.status:
            display_database_status(orchestrator, logger)
            return 0

        elif args.load_data:
            load_data_command(orchestrator, logger, max_articles=args.max_articles)
            return 0

        else:  # args.clear_db
            if args.force:
                # Skip confirmation
                logger.warning("Database clear requested with --force")
                result = orchestrator.clear_all_indices()
                if result['success']:
                    print(f"[OK] Success: {result['message']}")
                    logger.info("Database cleared")
                else:
                    print(f"[FAIL] Failed: {result['message']}")
                    logger.error(f"Database clear failed: {result['message']}")
                return 0 if result['success'] else 1
            else:
                clear_database_command(orchestrator, logger)
                return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n(Interrupted)")
        return 130

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
