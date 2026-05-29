"""Command-line wrapper to run RAG queries without pytest.

Examples:
    python src/rag_cli.py --query "What is LSI?"
    python src/rag_cli.py --interactive
    python src/rag_cli.py --query "Explain ChromaDB" --template chain_of_thought --log-llm
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List


def configure_logging(verbose: bool) -> None:
    """Configure logging for CLI execution."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_real_documents(data_file: Path) -> List[Dict]:
    """Load project documents from JSON file.

    Args:
        data_file: Path to JSON file with documents.

    Returns:
        List of documents.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file content is empty.
    """
    if not data_file.exists():
        raise FileNotFoundError(f"Documents file not found: {data_file}")

    with data_file.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    if not documents:
        raise ValueError(f"No documents found in: {data_file}")

    return documents


def ensure_pipeline_indexed(pipeline, data_file: Path) -> None:
    """Ensure SRIPipeline has searchable content using real project documents."""
    needs_indexing = (not pipeline.lsi.is_fitted) and (pipeline.vstore.count() == 0)
    if not needs_indexing:
        return

    docs = load_real_documents(data_file)
    pipeline.index(docs, save=False)


def to_rag_documents(results: List[Dict]) -> List[Dict]:
    """Convert pipeline search results into RAG document format."""
    return [
        {
            "id": item.get("doc_id") or item.get("id"),
            "title": item.get("title", "Untitled"),
            "content": item.get("content", ""),
            "url": item.get("url", ""),
            "score": item.get("score"),
        }
        for item in results
    ]


def maybe_wrap_llm_logging(provider, enabled: bool) -> None:
    """Wrap provider.generate to print full prompt and full model response."""
    if not enabled:
        return

    logger = logging.getLogger("rag_cli.llm")
    original_generate = provider.generate

    def logged_generate(prompt: str, **kwargs):
        logger.info("=== LLM PROMPT START ===\n%s\n=== LLM PROMPT END ===", prompt)
        response = original_generate(prompt=prompt, **kwargs)
        logger.info("=== LLM RESPONSE START ===\n%s\n=== LLM RESPONSE END ===", response)
        return response

    provider.generate = logged_generate


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run project RAG manually from CLI")
    parser.add_argument("--query", type=str, help="Single query to execute")
    parser.add_argument("--interactive", action="store_true", help="Interactive query loop")
    parser.add_argument("--model", default="llama3.2:latest", help="Ollama model name")
    parser.add_argument(
        "--template",
        default="domain_specific",
        choices=["basic", "domain_specific", "chain_of_thought"],
        help="Prompt template strategy",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retriever top-k documents")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="LLM max output tokens")
    parser.add_argument("--top-p", type=float, default=0.95, help="LLM top-p")
    parser.add_argument(
        "--data-file",
        default="data/documents.json",
        help="Path to real project documents JSON",
    )
    parser.add_argument("--log-llm", action="store_true", help="Log full prompts and responses")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def run_query(args, rag, pipeline, query: str) -> None:
    """Execute one query and print answer with citations."""
    results = pipeline.search(query, top_k=args.top_k)
    rag_docs = to_rag_documents(results)

    response = rag.generate(
        query,
        documents=rag_docs,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    print("\n=== ANSWER ===")
    print(response.answer)

    print("\n=== CITATIONS ===")
    if not response.citations:
        print("(none)")
        return

    for citation in response.citations:
        print(f"- {citation.doc_id} | {citation.title} | {citation.url}")


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("Use --query for one query or --interactive for chat mode")

    configure_logging(args.verbose)

    # Delayed imports keep --help usable even if optional deps are missing.
    from sri.pipeline import SRIPipeline
    from rag.rag_module import RAGModule
    from rag.llm_provider import OllamaProvider

    pipeline = SRIPipeline(load_existing=True)
    ensure_pipeline_indexed(pipeline, Path(args.data_file))

    provider = OllamaProvider(model=args.model)
    maybe_wrap_llm_logging(provider, args.log_llm)

    rag = RAGModule(llm=provider, template_type=args.template, pipeline=pipeline)

    if args.query:
        run_query(args, rag, pipeline, args.query)
        return 0

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        run_query(args, rag, pipeline, query)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
