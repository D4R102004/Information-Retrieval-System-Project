"""Recommendation tab for the Gradio UI."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ..services.orchestrator_service import get_orchestrator_service


def _parse_doc_ids(raw_ids: str) -> list[str]:
    """Parse comma/newline-separated document ids from the UI."""
    if not raw_ids:
        return []
    normalized = raw_ids.replace("\n", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _format_recommendations(result: dict[str, Any]) -> str:
    """Render recommendation results as Markdown cards."""
    status = result.get("status", "error")
    message = result.get("message", "")
    recommendations = result.get("recommendations") or []
    metadata = result.get("metadata") or {}

    if status == "empty":
        return f"### Recommendation results\n\nℹ️ {message or 'No search history available yet.'}"

    if status != "success":
        return f"### Recommendation results\n\n❌ {message or 'Recommendation failed.'}"

    queries_used = metadata.get("queries_used") or []
    history_note = ""
    if queries_used:
        recent = " | ".join(str(query) for query in queries_used[-5:])
        history_note = f"\n**Based on recent searches:** {recent}\n"

    lines = [
        "### Recommendation results",
        "",
        f"**{message}**",
        f"Corpus documents: `{metadata.get('total_documents', 0)}` | Candidates: `{metadata.get('candidate_documents', 0)}`",
        history_note,
        "",
    ]

    if not recommendations:
        lines.append("No recommendations were generated.")
        return "\n".join(lines)

    for index, doc in enumerate(recommendations, start=1):
        title = doc.get("title") or "Untitled document"
        url = doc.get("url") or ""
        doc_id = doc.get("id") or doc.get("doc_id") or ""
        source = doc.get("source") or "unknown"
        tags = doc.get("tags") or ""
        score = doc.get("recommendation_score", 0)
        sim = doc.get("similarity_score", 0)
        recency = doc.get("recency_score", 0)
        explanation = doc.get("explanation") or "Recommended by content similarity."
        snippet = doc.get("snippet") or ""

        rendered_title = f"[{title}]({url})" if url else str(title)
        lines.extend(
            [
                f"#### {index}. {rendered_title}",
                f"**Local ID**: `{doc_id}`  (intended for manual recommendations)",
                f"**Source:** {source}  ",
                f"**Score:** {score} | **Similarity:** {sim} | **Recency:** {recency} ",
                f"**Tags:** {tags}" if tags else "**Tags:** — ",
                f"**Why:** {explanation} ",
                f"> {snippet[:450]} " if snippet else "",
                "---",
            ]
        )

    return "\n".join(lines)


def build_recommendation_tab() -> gr.Markdown:
    """Build the recommendation module tab."""
    service = get_orchestrator_service()

    gr.Markdown(
        """
        ## Recommendation module

        Generate recommendations from the local corpus. The automatic section updates after searches
        from the Search tab and uses at least the latest 5 stored searches when available.
        """
    )

    gr.Markdown("### Automatic recommendations from recent searches")
    with gr.Row():
        auto_top_k_input = gr.Slider(
            label="Automatic results",
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            scale=2,
        )
        auto_recommend_button = gr.Button("Generate from latest 5 searches", variant="primary", scale=3)
        clear_history_button = gr.Button("Clear search history", variant="secondary", scale=2)

    automatic_recommendation_output = gr.Markdown(
        "### Automatic recommendations\n\nRun a search in the Search tab to generate recommendations from your latest searches."
    )

    gr.Markdown("### Manual recommendations")
    with gr.Row():
        with gr.Column(scale=5):
            interests_input = gr.Textbox(
                label="User interests / profile",
                placeholder="Example: Python, APIs, cloud computing, LLM tools, cybersecurity...",
                lines=3,
            )
            query_input = gr.Textbox(
                label="Optional current query/topic",
                placeholder="Example: serverless real-time applications",
                lines=1,
            )
            liked_ids_input = gr.Textbox(
                label="Liked or seed document IDs",
                placeholder="Paste document IDs separated by commas or new lines",
                lines=4,
            )
            top_k_input = gr.Slider(
                label="Number of recommendations",
                minimum=1,
                maximum=20,
                value=10,
                step=1,
            )
            with gr.Row():
                recommend_button = gr.Button("Generate recommendations", variant="primary")
                refresh_button = gr.Button("Refresh recommendation model")

        with gr.Column(scale=7):
            recommendation_output = gr.Markdown(
                "### Recommendation results\n\nNo recommendation has been generated yet."
            )

    gr.Markdown("### Similar documents")
    with gr.Row():
        similar_doc_input = gr.Textbox(
            label="Document ID",
            placeholder="Paste one document ID to find similar articles",
            scale=5,
        )
        similar_top_k_input = gr.Slider(
            label="Results",
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            scale=2,
        )
        similar_button = gr.Button("Find similar", variant="secondary", scale=2)

    similar_output = gr.Markdown("### Similar documents\n\nNo document selected yet.")

    def run_recommendation(interests: str, query: str, liked_ids: str, top_k: int) -> str:
        result = service.recommend_documents(
            query=query,
            interests=interests,
            liked_doc_ids=_parse_doc_ids(liked_ids),
            top_k=int(top_k),
        )
        return _format_recommendations(result)

    def refresh_model() -> str:
        result = service.refresh_recommender()
        icon = "✅" if result.get("success") else "❌"
        return (
            "### Recommendation results\n\n"
            f"{icon} {result.get('message', 'Refresh completed.')}"
        )

    def run_automatic_recommendation(top_k: int) -> str:
        result = service.recommend_from_history(top_k=int(top_k), history_limit=5)
        text = _format_recommendations(result)
        return text.replace("### Recommendation results", "### Automatic recommendations", 1)

    def clear_history() -> str:
        result = service.clear_search_history()
        icon = "✅" if result.get("status") == "success" else "❌"
        return (
            "### Automatic recommendations\n\n"
            f"{icon} {result.get('message', 'Search history cleared.')}"
        )

    def run_similar(document_id: str, top_k: int) -> str:
        if not document_id or not document_id.strip():
            return "### Similar documents\n\nPlease provide a document ID."
        result = service.recommend_similar_documents(document_id.strip(), top_k=int(top_k))
        text = _format_recommendations(result)
        return text.replace("### Recommendation results", "### Similar documents", 1)

    auto_recommend_button.click(
        fn=run_automatic_recommendation,
        inputs=[auto_top_k_input],
        outputs=[automatic_recommendation_output],
    )
    clear_history_button.click(fn=clear_history, inputs=[], outputs=[automatic_recommendation_output])

    recommend_button.click(
        fn=run_recommendation,
        inputs=[interests_input, query_input, liked_ids_input, top_k_input],
        outputs=[recommendation_output],
    )
    refresh_button.click(fn=refresh_model, inputs=[], outputs=[recommendation_output])
    similar_button.click(
        fn=run_similar,
        inputs=[similar_doc_input, similar_top_k_input],
        outputs=[similar_output],
    )

    return automatic_recommendation_output
