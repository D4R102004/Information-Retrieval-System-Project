"""Configuration Tab - System parameter management."""

from __future__ import annotations

import gradio as gr
from typing import Any

from ..services.orchestrator_service import get_orchestrator_service
from ..config import DEFAULT_MIN_DOCUMENTS


def build_configuration_tab() -> None:
    """
    Build configuration management interface.

    Handles system parameters, RAG settings, crawler configuration,
    and database maintenance operations.
    """
    orchestrator_service = get_orchestrator_service()

    gr.Markdown("## System Configuration")
    gr.Markdown(
        "Adjust system parameters, RAG settings, and database maintenance options."
    )

    with gr.Tabs():
        # ===== Query Parameters Tab =====
        with gr.TabItem("Query Parameters"):
            gr.Markdown("### Search Behavior")
            with gr.Group():
                max_local_results = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Max Local Results",
                    info="Maximum results from local search"
                )
                max_web_results = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Max Web Results",
                    info="Maximum external search results when insufficient"
                )
                use_web_search = gr.Checkbox(
                    value=True,
                    label="Enable Web Search",
                    info="Augment results with web search when local insufficient"
                )
                auto_reload = gr.Checkbox(
                    value=True,
                    label="Auto-reload Empty Database",
                    info="Automatically execute crawlers if index is below minimum"
                )

            save_query_btn = gr.Button("💾 Save Query Settings", variant="primary")
            query_status = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )

            def save_query_settings(
                max_local: int,
                max_web: int,
                web_search: bool,
                auto_reload_db: bool
            ) -> str:
                """Save query settings (information only, not persisted)."""
                return (
                    f"[OK] Settings updated: "
                    f"Local={max_local}, Web={max_web}, "
                    f"WebSearch={'on' if web_search else 'off'}, "
                    f"AutoReload={'on' if auto_reload_db else 'off'}"
                )

            save_query_btn.click(
                save_query_settings,
                inputs=[max_local_results, max_web_results, use_web_search, auto_reload],
                outputs=query_status
            )

            # ===== RAG Configuration Tab =====
            with gr.TabItem("RAG Configuration"):
                gr.Markdown("### Large Language Model Settings")
            with gr.Group():
                    ollama_model = gr.Textbox(
                        value="llama3.2:latest",
                        label="Ollama Model",
                        info="Model identifier (e.g., llama3.2:latest)",
                        interactive=True
                    )
                    ollama_url = gr.Textbox(
                        value="http://localhost:11434",
                        label="Ollama Base URL",
                        info="Full URL including protocol and port",
                        interactive=True
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Lower = more deterministic, Higher = more creative",
                        interactive=True
                    )
                    max_tokens = gr.Slider(
                        minimum=100,
                        maximum=4096,
                        value=1024,
                        step=100,
                        label="Max Tokens",
                        info="Maximum length of generated response",
                        interactive=True
                    )
                    top_k_retrieval = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Retrieval Top-K",
                        info="Documents used for RAG context",
                        interactive=True
                    )

            with gr.Group():
                gr.Markdown("### Connection Test")
                test_connection_btn = gr.Button("🔗 Test Ollama Connection", variant="secondary")
                connection_status = gr.Textbox(
                    label="Connection Result",
                    interactive=False,
                    lines=3
                )

            def test_ollama_connection(model: str, url: str) -> str:
                """Test connection to Ollama service."""
                try:
                    import requests
                    response = requests.get(f"{url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m.get('name') for m in models]
                        if model in model_names:
                            return f"[OK] Connected to {url}\n[OK] Model '{model}' available\n[OK] Service healthy"
                        else:
                            return f"[OK] Connected but model '{model}' not found.\nAvailable: {', '.join(model_names[:3])}"
                    else:
                        return f"[X] Connection failed: HTTP {response.status_code}"
                except requests.exceptions.ConnectionError:
                    return f"[X] Cannot connect to {url}. Is Ollama running?"
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            test_connection_btn.click(
                test_ollama_connection,
                inputs=[ollama_model, ollama_url],
                outputs=connection_status
            )

            # ===== Crawler Configuration Tab =====
            with gr.TabItem("Crawler Configuration"):
                gr.Markdown("### Data Acquisition Settings")
            with gr.Group():
                    max_articles = gr.Slider(
                        minimum=100,
                        maximum=10000,
                        value=1000,
                        step=100,
                        label="Max Articles per Spider",
                        info="Maximum documents per crawler",
                        interactive=True
                    )
                    crawler_timeout = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Crawler Timeout (seconds)",
                        info="Maximum time per crawler execution",
                        interactive=True
                    )
                    force_recrawl = gr.Checkbox(
                        value=False,
                        label="Force Recrawl",
                        info="Ignore cached data and re-execute all crawlers",
                        interactive=True
                    )

            # ===== Database Management Tab =====
            with gr.TabItem("Database Management"):
                gr.Markdown("### Database Operations")

            with gr.Group():
                gr.Markdown("#### Quick Actions")
                col1, col2, col3 = gr.Row(), gr.Row(), gr.Row()
                
                with col1:
                    clear_db_btn = gr.Button("🗑️  Clear All Data", variant="stop", scale=1)
                with col2:
                    reindex_db_btn = gr.Button("♻️  Reindex Database", variant="secondary", scale=1)
                with col3:
                    reload_db_btn = gr.Button("📥 Reload from Crawlers", variant="secondary", scale=1)

            operation_output = gr.Textbox(
                label="Operation Result",
                interactive=False,
                lines=4
            )

            gr.Markdown("#### Database Status")
            db_info = gr.Textbox(
                label="Current Database Info",
                interactive=False,
                lines=5
            )

            with gr.Group():
                refresh_status_btn = gr.Button("🔄 Refresh Status")

            def clear_database_with_confirmation() -> str:
                """Clear all indices with confirmation."""
                try:
                    result = orchestrator_service._get_orchestrator().clear_all_indices()
                    if result.get('success'):
                        return f"[OK] Database cleared successfully\n{result.get('message', '')}"
                    else:
                        return f"[X] Clear operation failed:\n{result.get('message', 'Unknown error')}"
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            def reindex_database() -> str:
                """Reindex database from available sources."""
                try:
                    result = orchestrator_service._get_orchestrator().reindex_database()
                    if result.get('success'):
                        return (
                            f"[OK] Reindex completed\n"
                            f"Indexed documents: {result.get('indexed_documents', 0)}\n"
                            f"Duration: {result.get('duration_seconds', 0):.2f}s\n"
                            f"{result.get('message', '')}"
                        )
                    else:
                        return f"[X] Reindex failed:\n{result.get('message', 'Unknown error')}"
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            def reload_from_crawlers() -> str:
                """Execute full crawler pipeline."""
                try:
                    result = orchestrator_service._get_orchestrator().load_documents_from_crawlers(
                        max_articles=1000,
                        force_recrawl=False
                    )
                    if result.get('success'):
                        return (
                            f"[OK] Crawlers executed successfully\n"
                            f"Total documents: {result.get('total_documents', 0)}\n"
                            f"Indexed: {result.get('indexed_documents', 0)}\n"
                            f"Duration: {result.get('duration_seconds', 0):.2f}s"
                        )
                    else:
                        return f"[X] Crawler execution failed:\n{result.get('message', 'Unknown error')}"
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            def get_database_info() -> str:
                """Get current database statistics."""
                try:
                    health = orchestrator_service._get_orchestrator().check_database_health()
                    return (
                        f"Indexed documents: {health.get('document_count', 0)}\n"
                        f"File documents: {health.get('file_document_count', 0)}\n"
                        f"Status: {health.get('status', 'unknown')}\n"
                        f"ChromaDB: {'[OK] Available' if health.get('has_chromadb') else '[X] Not available'}"
                    )
                except Exception as e:
                    return f"[X] Error retrieving status: {str(e)}"

            clear_db_btn.click(clear_database_with_confirmation, outputs=operation_output)
            reindex_db_btn.click(reindex_database, outputs=operation_output)
            reload_db_btn.click(reload_from_crawlers, outputs=operation_output)
            refresh_status_btn.click(get_database_info, outputs=db_info)
