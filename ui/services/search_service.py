"""Search-related helpers for the frontend."""

from __future__ import annotations

from typing import Any
from math import log10 as log
import re

from src.rag.output_parser import RAGResponse, CitationExtractor

CITATION_PATTERN = CitationExtractor.CITATION_PATTERN
CODE_BLOCK_PATTERN = CitationExtractor.CODE_BLOCK_PATTERN


def map_progress_event(event: dict[str, Any]) -> dict[str, str] | None:
    """Map backend log/status events to compact user-facing progress states."""
    kind = str(event.get("kind", "")).lower()
    message = str(event.get("message", "")).strip()
    lowered = message.lower()

    if kind == "status":
        return {
            "stage": "checking",
            "label": "Checking local database",
            "detail": message or "Starting retrieval workflow",
        }

    if kind != "log" or not message:
        return None

    if "db counts ->" in lowered or "retrieving documents for" in lowered:
        return {
            "stage": "checking",
            "label": "Checking local database",
            "detail": "Validating available indexed documents",
        }

    if "consolidating raw documents" in lowered:
        return {
            "stage": "consolidating",
            "label": "Consolidating raw data",
            "detail": "Preparing a unified document set from raw sources",
        }

    if "consolidated and indexed" in lowered or "indexando" in lowered or "indexed " in lowered:
        return {
            "stage": "indexing",
            "label": "Building data base",
            "detail": "Updating local indexes for faster retrieval",
        }

    if "performing web search" in lowered or "web search returned" in lowered:
        return {
            "stage": "web_search",
            "label": "Looking for documents online",
            "detail": "Collecting complementary sources from the web",
        }

    if "consolidated " in lowered and "documents" in lowered:
        return {
            "stage": "finalizing",
            "label": "Finalizing retrieved documents",
            "detail": "Preparing results for the response panel",
        }

    return None


def format_progress_panel(progress_events: list[dict[str, str]]) -> str:
    """Render user-friendly progress timeline as Markdown."""
    if not progress_events:
        return (
            "### Query progress\n\n"
            "No query in progress."
        )

    latest = progress_events[-1]
    lines = [
        "### Query progress",
        f"**Current status:** {latest.get('label', 'Working...')}",
    ]

    detail = latest.get("detail", "")
    if detail:
        lines.append(f"{detail}")

    lines.append("")
    lines.append("**Recent steps**")
    for event in progress_events[-4:]:
        lines.append(f"- {event.get('label', 'Working...')}")

    return "\n\n".join(lines)


def build_result_payload(
    documents: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Shape retrieval output for the UI layer."""
    return {"documents": documents, "metadata": metadata}


def format_search_results(
    documents: list[dict[str, Any]], metadata: dict[str, Any]
) -> str:
    """Render retrieved documents as concise Markdown."""
    if not documents:
        return (
            "### Retrieval results\n\n"
            "No documents were retrieved."
        )

    lines = [
        "### Retrieval results",
        f"Retrieved **{len(documents)}** documents.",
    ]

    local_count = metadata.get("local_documents", 0)
    web_count = metadata.get("web_documents", 0)
    total_count = metadata.get("total_documents_used", len(documents))
    lines.append(
        f"- Local: **{local_count}** | Web: **{web_count}** | Total: **{total_count}**"
    )

    if metadata.get("insufficiency_detected"):
        reasons = metadata.get("insufficiency_reasons", [])
        if reasons:
            reason_text = ", ".join(str(reason) for reason in reasons)
            lines.append(f"- Insufficiency: {reason_text}")

    lines.append("")
    for index, document in enumerate(documents, 1):
        title = document.get("title") or document.get("id") or f"Document {index}"
        source = document.get("source") or "local"
        score = document.get("score")
        url = document.get("url")
        snippet = document.get("content") or document.get("snippet") or "No snippet available."
        snippet = " ".join(str(snippet).split())
        if len(snippet) > 220:
            snippet = f"{snippet[:217]}..."

        lines.append(f"#### {index}. {title}")
        score_text = f"Score: {(100*score):.2f}%" if score else ""
        source_text = f"Source: {source}" if source else ""
        if score_text or source_text:
            separator = " | " if score_text and source_text else ""
            lines.append(f"- {source_text}{separator}{score_text}")
        if url:
            lines.append(f"- URL: {url}")
        lines.append(f"- Snippet: {snippet}")
        lines.append("")

    return "\n".join(lines).strip()


def format_rag_response(response: RAGResponse) -> str:
    """Render the generated answer and citations as Markdown."""
    lines = ["### RAG answer"]
    answer_provided = response.answer and response.answer.strip() != ""

    citations = response.citations if response.citations else []

    citation_map = []
    for index, citation in enumerate(citations, 1):
        citation_id = getattr(citation, 'doc_id', None) or getattr(citation, 'id', None)
        if citation_id:
            citation_map.append((str(citation_id), index))

    shown_answer = "No answer was generated."
    if answer_provided:
        shown_answer = response.answer.strip()
        #TODO: Remove on production - this is only to debug the raw answer before citation mapping
        lines.append("#### RAW answer")
        lines.append(shown_answer)
        lines.append("")

    for citation_id, index in sorted(citation_map, key=lambda item: len(item[0]), reverse=True):
        shown_answer = re.sub(rf"\[{re.escape(citation_id)}\]", f"[{index}]", shown_answer)
    

    # Remove code blocks to avoid identifying lists as false citations
    no_code_answer = re.sub(CODE_BLOCK_PATTERN, "", shown_answer)
    
    # Build a set of current ids mapped to indices
    mapped_ids = {idx for _, idx in citation_map}

    # RAG will never receive 10^5 doc ids, so this is a safe heuristic to avoid showing 
    # false citations that are actually part of the answer text.
    unremoved_citations = [f"[{cite}]" 
                           for cite in re.findall(CITATION_PATTERN, no_code_answer) 
                           if len(cite) > (log(len(mapped_ids)) + 1) and cite not in mapped_ids
                           ]

    for unremoved in unremoved_citations:
        shown_answer = shown_answer.replace(unremoved, "")
    
    lines.append("#### Final answer")
    lines.append(shown_answer)

    if answer_provided and len(citations) > 0:
        lines.append("")
        lines.append("### Citations")

        for index, citation in enumerate(citations, 1):
            title = citation.title or "Untitled"
            url = citation.url or ""
            snippet = citation.snippet or ""
            date = citation.date or ""

            lines.append(f"[{index}] **{title}**")
            if url:
                lines.append(f"   - URL: {url}")
            if date:
                lines.append(f"   - Date: {date}")
            if snippet:
                lines.append(f"   - Snippet: {snippet}")
            
            lines.append("")
    elif answer_provided:
        lines.append("")
        lines.append("### Citations")
        lines.append("No citations were extracted.")
    else:
        lines.append("No answer was generated.")

    return "\n".join(lines).strip()


def format_search_status(metadata: dict[str, Any], last_query: str) -> str:
    """Render the compact query footer shown below the search controls."""
    lines = [
        f"**Minimum required documents:** {metadata.get('minimum_documents', 500)}",
        f"**Last query:** {last_query or 'No query yet'}",
        f"**Local results:** {metadata.get('local_documents', 0)} | **Web results:** {metadata.get('web_documents', 0)}",
    ]

    reasons = metadata.get("insufficiency_reasons", [])
    if reasons:
        lines.append(f"**Insufficiency:** {', '.join(str(reason) for reason in reasons)}")

    return "\n\n".join(lines)
