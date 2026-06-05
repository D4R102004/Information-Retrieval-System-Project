"""Gradio application entry point for the frontend."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .config import APP_SUBTITLE, APP_THEME_CSS, APP_TITLE, DEFAULT_MIN_DOCUMENTS
from .services.orchestrator_service import get_orchestrator_service
from .services.search_service import (
    format_progress_panel,
    format_rag_response,
    format_search_results,
    format_search_status,
    map_progress_event,
)
from .state import UIState, create_default_state
from .tabs.configuration import build_configuration_tab
from .tabs.evaluation import build_evaluation_tab
from .tabs.recommendation import build_recommendation_tab, _format_recommendations
from .tabs.status import build_status_tab
from .utils import validate_query


def _load_theme_css() -> str:
    """Load the shared CSS file if it exists."""
    css_path = Path(__file__).parent / APP_THEME_CSS
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


def _render_placeholder_results(state: UIState) -> tuple[str, str, str, str]:
    """Render placeholder content for the search and status panels."""
    results = (
        "### Retrieval results\n\n"
        "No query has been executed yet.\n\n"
        "This space will list documents, snippets, and metadata once retrieval is connected."
    )
    progress = "### Query progress\n\nNo query in progress."
    rag_panel = (
        "### RAG panel\n\nRAG output will appear here after retrieval completes."
    )
    status_panel = format_search_status(
        {
            "minimum_documents": state.settings.get("min_documents", DEFAULT_MIN_DOCUMENTS),
            "local_documents": 0,
            "web_documents": 0,
            "insufficiency_reasons": [],
        },
        state.last_query,
    )
    return results, progress, rag_panel, status_panel


def create_app() -> gr.Blocks:
    """Create the Gradio frontend shell."""
    state = create_default_state()
    orchestrator_service = get_orchestrator_service()

    with gr.Blocks(title=APP_TITLE) as demo:
        app_state = gr.State(state)

        gr.HTML(
            f"""
            <div class="ui-hero">
              <h1>{APP_TITLE}</h1>
              <p>{APP_SUBTITLE}</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Search"):
                with gr.Row(elem_classes=["ui-search-layout"]):
                    with gr.Column(scale=4, elem_classes=["ui-query-column"]):
                        query_input = gr.Textbox(
                            label="Search query",
                            placeholder="Enter your search query...",
                            lines=1,
                            max_lines=2,
                        )
                        with gr.Row():
                            search_button = gr.Button("Search", variant="primary")
                            clear_button = gr.Button("Clear")

                        status_output = gr.Markdown(elem_classes=["ui-query-status"])

                    with gr.Column(scale=8, elem_classes=["ui-results-column"]):
                        with gr.Row(elem_classes=["ui-results-row"]):
                            with gr.Column(scale=8, elem_classes=["ui-results-panel"]):
                                results_output = gr.Markdown()
                            with gr.Column(scale=4, elem_classes=["ui-rag-panel"]):
                                rag_output = gr.Markdown()
                        progress_output = gr.Markdown(
                            elem_classes=["ui-progress-panel"]
                        )

                def run_search(
                    query: str,
                    session_state: UIState,
                ):
                    """Retrieve documents and generate RAG response using settings from state."""
                    progress_events: list[dict[str, str]] = []

                    settings = session_state.settings
                    web_search_enabled = settings.get("enable_web_search", True)
                    auto_reload_enabled = settings.get("auto_reload", True)
                    local_results = settings.get("max_local_results", 5)
                    is_valid, error_message = validate_query(query)
                    if not is_valid:
                        session_state.last_query = query or ""
                        session_state.retrieved_documents = []
                        session_state.rag_response = {}
                        progress_events.append(
                            {
                                "stage": "validation_error",
                                "label": "Invalid query",
                                "detail": error_message,
                            }
                        )
                        yield (
                            error_message,
                            "",
                            format_progress_panel(progress_events),
                            error_message,
                            session_state,
                            "",
                        )
                        return

                    session_state.last_query = query.strip()
                    retrieval_result: dict[str, object] = {}
                    for (
                        retrieval_event
                    ) in orchestrator_service.stream_retrieve_documents(
                        question=session_state.last_query,
                        max_local_results=local_results,
                        enable_web_search=web_search_enabled,
                        auto_reload=auto_reload_enabled,
                    ):
                        if retrieval_event.get("kind") == "result":
                            retrieval_result = retrieval_event.get("payload", {}) or {}
                            continue

                        progress_event = map_progress_event(retrieval_event)
                        if not progress_event:
                            continue

                        if progress_events and progress_events[-1].get(
                            "stage"
                        ) == progress_event.get("stage"):
                            progress_events[-1] = progress_event
                        else:
                            progress_events.append(progress_event)

                        yield (
                            "### Retrieval results\n\nCollecting documents...",
                            "### RAG panel\n\nWaiting for retrieval to finish.",
                            format_progress_panel(progress_events),
                            format_search_status(
                                {"minimum_documents": DEFAULT_MIN_DOCUMENTS},
                                session_state.last_query,
                            ),
                            session_state,
                            "",
                        )

                    documents_raw = retrieval_result.get("documents", [])
                    metadata_raw = retrieval_result.get("metadata", {})
                    error_raw = retrieval_result.get("error")

                    documents: list[dict[str, object]] = (
                        documents_raw if isinstance(documents_raw, list) else []
                    )
                    metadata: dict[str, object] = (
                        metadata_raw if isinstance(metadata_raw, dict) else {}
                    )
                    error_message = str(error_raw) if error_raw else ""

                    if error_message:
                        session_state.retrieved_documents = []
                        session_state.rag_response = {}
                        status_text = format_search_status(
                            metadata, session_state.last_query
                        )
                        progress_events.append(
                            {
                                "stage": "retrieval_error",
                                "label": "Retrieval failed",
                                "detail": str(error_message),
                            }
                        )
                        yield (
                            error_message,
                            "",
                            format_progress_panel(progress_events),
                            status_text,
                            session_state,
                            "",
                        )
                        return

                    session_state.retrieved_documents = [
                        doc for doc in documents if isinstance(doc, dict)
                    ]
                    results_text = format_search_results(documents, metadata)
                    status_text = format_search_status(
                        metadata, session_state.last_query
                    )

                    loading_rag = (
                        "### RAG answer\n\n"
                        "Generating answer...\n\n"
                        "### Citations\n\nPending retrieval of the generated response."
                    )

                    progress_events.append(
                        {
                            "stage": "rag_generation",
                            "label": "Generating final answer",
                            "detail": "Composing response and extracting citations",
                        }
                    )

                    yield (
                        results_text,
                        loading_rag,
                        format_progress_panel(progress_events),
                        status_text,
                        session_state,
                        "",
                    )

                    rag_response = orchestrator_service.augment_response(
                        question=session_state.last_query,
                        documents=documents,
                    )
                    session_state.rag_response = rag_response.model_dump()

                    progress_events.append(
                        {
                            "stage": "completed",
                            "label": "Completed",
                            "detail": "Results and answer are ready",
                        }
                    )

                    automatic_recommendation = orchestrator_service.recommend_from_history(
                        top_k=10,
                        history_limit=5,
                    )
                    automatic_recommendation_text = _format_recommendations(automatic_recommendation).replace(
                        "### Recommendation results",
                        "### Automatic recommendations",
                        1,
                    )

                    yield (
                        results_text,
                        format_rag_response(rag_response),
                        format_progress_panel(progress_events),
                        status_text,
                        session_state,
                        automatic_recommendation_text,
                    )

                def reset_search(
                    session_state: UIState,
                ) -> tuple[str, str, str, str, str, UIState]:
                    """Reset the search tab to its default empty state."""
                    default_state = create_default_state()
                    results_text, progress_text, rag_text, status_text = (
                        _render_placeholder_results(default_state)
                    )
                    return (
                        "",
                        results_text,
                        progress_text,
                        rag_text,
                        status_text,
                        default_state,
                    )

                clear_button.click(
                    fn=reset_search,
                    inputs=[app_state],
                    outputs=[
                        query_input,
                        results_output,
                        progress_output,
                        rag_output,
                        status_output,
                        app_state,
                    ],
                )

                default_results, default_progress, default_rag, default_status = (
                    _render_placeholder_results(state)
                )
                results_output.value = default_results
                progress_output.value = default_progress
                rag_output.value = default_rag
                status_output.value = default_status

            with gr.Tab("Configuration"):
                build_configuration_tab(app_state)

            with gr.Tab("Evaluation"):
                build_evaluation_tab()

            with gr.Tab("Recommendation"):
                automatic_recommendation_output = build_recommendation_tab()

            with gr.Tab("System Status"):
                build_status_tab()


        search_button.click(
            fn=run_search,
            inputs=[query_input, app_state],
            outputs=[
                results_output,
                rag_output,
                progress_output,
                status_output,
                app_state,
                automatic_recommendation_output,
            ],
        )

        demo.load(
            fn=lambda current_state: _render_placeholder_results(current_state),
            inputs=[app_state],
            outputs=[results_output, progress_output, rag_output, status_output],
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""
    demo = create_app()
    demo.launch(css=_load_theme_css())


if __name__ == "__main__":
    main()
