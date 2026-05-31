"""Gradio application entry point for the frontend."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .config import APP_SUBTITLE, APP_THEME_CSS, APP_TITLE, DEFAULT_MIN_DOCUMENTS
from .state import UIState, create_default_state, state_to_dict
from .utils import build_status_message, validate_query


def _load_theme_css() -> str:
    """Load the shared CSS file if it exists."""
    css_path = Path(__file__).parent / APP_THEME_CSS
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


def _render_placeholder_results(state: UIState) -> tuple[str, str, str]:
    """Render placeholder content for the search and status panels."""
    results = "\n".join(
        [
            "### Retrieval results",
            "No query has been executed yet.",
            "",
            "This space will list documents, snippets, and metadata once retrieval is connected.",
        ]
    )
    rag_panel = "### RAG panel\n\nRAG output will appear here after retrieval completes."
    status_panel = "\n\n".join(
        [
            build_status_message("Minimum indexed documents required", DEFAULT_MIN_DOCUMENTS),
            build_status_message("Last query", state.last_query or "No query yet"),
        ]
    )
    return results, rag_panel, status_panel


def create_app() -> gr.Blocks:
    """Create the Gradio frontend shell."""
    theme_css = _load_theme_css()
    state = create_default_state()

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

                        with gr.Accordion("Search options", open=False):
                            use_web_search = gr.Checkbox(value=True, label="Use web search")
                            auto_reload = gr.Checkbox(value=True, label="Auto-reload database")
                            max_local_results = gr.Slider(
                                1,
                                20,
                                value=5,
                                step=1,
                                label="Max local results",
                            )
                            max_web_results = gr.Slider(
                                0,
                                20,
                                value=10,
                                step=1,
                                label="Max web results",
                            )

                        status_output = gr.Markdown(elem_classes=["ui-query-status"])

                    with gr.Column(scale=8, elem_classes=["ui-results-column"]):
                        with gr.Row(elem_classes=["ui-results-row"]):
                            with gr.Column(scale=8, elem_classes=["ui-results-panel"]):
                                results_output = gr.Markdown()
                            with gr.Column(scale=4, elem_classes=["ui-rag-panel"]):
                                rag_output = gr.Markdown()

                def run_search(
                    query: str,
                    web_search_enabled: bool,
                    auto_reload_enabled: bool,
                    local_results: int,
                    web_results: int,
                    session_state: UIState,
                ) -> tuple[str, str, str, UIState]:
                    """Validate the query and update the visible shell state."""
                    is_valid, error_message = validate_query(query)
                    if not is_valid:
                        session_state.last_query = query or ""
                        session_state.retrieved_documents = []
                        session_state.rag_response = {}
                        return error_message, "", error_message, session_state

                    session_state.last_query = query.strip()
                    session_state.settings = {
                        "use_web_search": web_search_enabled,
                        "auto_reload_empty": auto_reload_enabled,
                        "max_local_results": local_results,
                        "max_web_results": web_results,
                    }
                    results_text, rag_text, status_text = _render_placeholder_results(session_state)
                    return results_text, rag_text, status_text, session_state

                def reset_search(session_state: UIState) -> tuple[str, str, str, str, UIState]:
                    """Reset the search tab to its default empty state."""
                    default_state = create_default_state()
                    results_text, rag_text, status_text = _render_placeholder_results(default_state)
                    return (
                        "",
                        results_text,
                        rag_text,
                        status_text,
                        default_state,
                    )

                search_button.click(
                    fn=run_search,
                    inputs=[
                        query_input,
                        use_web_search,
                        auto_reload,
                        max_local_results,
                        max_web_results,
                        app_state,
                    ],
                    outputs=[results_output, rag_output, status_output, app_state],
                )
                clear_button.click(
                    fn=reset_search,
                    inputs=[app_state],
                    outputs=[query_input, results_output, rag_output, status_output, app_state],
                )

                default_results, default_rag, default_status = _render_placeholder_results(state)
                results_output.value = default_results
                rag_output.value = default_rag
                status_output.value = default_status

            with gr.Tab("Configuration"):
                gr.Markdown("### Query parameters")
                gr.Markdown(
                    "This section will expose the settings described in Section 4.2, 4.3, 4.4, and 4.5 of the implementation plan."
                )
                gr.Markdown("### Database management")
                gr.Markdown("Clear, load, and health-check actions will be wired in step 3.")

            with gr.Tab("Evaluation"):
                gr.Markdown("### Evaluation workspace")
                gr.Markdown(
                    "This tab will host the default test execution path and the custom test designer described in Section 5."
                )

            with gr.Tab("System Status"):
                gr.Markdown("### System diagnostics")
                gr.Markdown(
                    "This tab will summarize database counts, crawler status, and LLM connectivity as defined in Section 6."
                )

        demo.load(
            fn=lambda current_state: _render_placeholder_results(current_state),
            inputs=[app_state],
            outputs=[results_output, rag_output, status_output],
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""
    demo = create_app()
    demo.launch(css=_load_theme_css())


if __name__ == "__main__":
    main()
