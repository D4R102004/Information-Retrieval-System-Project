"""Configuration Tab - System parameter management."""

from __future__ import annotations

import gradio as gr

from ..services.orchestrator_service import get_orchestrator_service
from ..state import UIState

from ..config import (
    DEFAULT_AUTO_RELOAD,
    DEFAULT_ENABLE_WEB_SEARCH,
    DEFAULT_MAX_LOCAL_RESULTS,
    DEFAULT_MAX_WEB_RESULTS,
    DEFAULT_USE_INITIAL_CORPUS,
    DEFAULT_MIN_DOCUMENTS,
    DEFAULT_QUERY_MIN_LENGTH,
    DEFAULT_QUERY_MAX_LENGTH,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RAG_TEMPLATE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RAG_TEMPERATURE,
    DEFAULT_RAG_MAX_TOKENS,
    DEFAULT_MAX_CITES,
    DEFAULT_MAX_CONTEXT_DOC_LENGTH,
    DEFAULT_MAX_ARTICLES_PER_SPIDER,
    DEFAULT_FORCE_RECRAWL,
    DEFAULT_CLEAR_RAW
)

def _to_snake_case(name: str) -> str:
    """Convert ['Basic', 'Domain Specific'] -> ['basic', 'domain_specific']"""
    return name.strip().lower().replace(" ", "_")

def _to_title_case(name: str) -> str:
    """Convert ['basic', 'domain_specific'] -> ['Basic', 'Domain Specific']"""
    return " ".join(word.capitalize() for word in name.split("_"))

def build_configuration_tab(app_state: gr.State) -> None:
    """Build configuration management interface."""
    orchestrator_service = get_orchestrator_service()

    gr.Markdown("## System Configuration")
    gr.Markdown(
        "Adjust system parameters, RAG settings, and database maintenance options."
    )

    with gr.Tabs():
        with gr.TabItem("Query Parameters"):
            gr.Markdown("### Search Behavior")
            with gr.Group():
                max_local_results = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=DEFAULT_MAX_LOCAL_RESULTS,
                    step=1,
                    label="Max Local Results",
                    info="Maximum results from local search",
                )
                max_web_results = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=DEFAULT_MAX_WEB_RESULTS,
                    step=1,
                    label="Max Web Results",
                    info="Maximum external search results when insufficient",
                )
                min_documents = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=DEFAULT_MIN_DOCUMENTS,
                    step=10,
                    label="Min Documents",
                    info="Minimun documents required in the Data Base to perform a search",
                )
                enable_web_search = gr.Checkbox(
                    value=DEFAULT_ENABLE_WEB_SEARCH,
                    label="Enable Web Search",
                    info="Augment results with web search when local insufficient",
                )
                auto_reload = gr.Checkbox(
                    value=DEFAULT_AUTO_RELOAD,
                    label="Auto-reload Empty Database",
                    info="Automatically execute crawlers if index is below minimum",
                )
            with gr.Group():    
                save_query_btn = gr.Button("💾 Save Query Settings", variant="primary")
                query_status = gr.Textbox(label="Status", interactive=False, lines=1)

            def save_query_settings(
                max_local: int,
                max_web: int,
                min_docs: int,
                web_search: bool,
                auto_reload_db: bool,
                session_state: UIState,
            ) -> tuple[str, UIState]:
                """Persist query settings into session state for use by Search tab."""
                session_state.settings["enable_web_search"] = web_search
                session_state.settings["auto_reload"] = auto_reload_db
                session_state.settings["max_local_results"] = max_local
                session_state.settings["max_web_results"] = max_web
                session_state.settings["min_documents"] = min_docs
                orchestrator_service.sync_backend(session_state.settings)
                
                status = (
                    f"[OK] Settings saved — \n"
                    f"Local={max_local}, Web={max_web}, MinRequired={min_docs}, \n"
                    f"WebSearch={'on' if web_search else 'off'}, \n"
                    f"AutoReload={'on' if auto_reload_db else 'off'}"
                )
                return status, session_state

            save_query_btn.click(
                save_query_settings,
                inputs=[
                    max_local_results,
                    max_web_results,
                    min_documents,
                    enable_web_search,
                    auto_reload,
                    app_state,
                ],
                outputs=[query_status, app_state],
            )

        with gr.TabItem("RAG Configuration"):
            gr.Markdown("### Large Language Model Settings")
            with gr.Group():
                ollama_model = gr.Textbox(
                    value=DEFAULT_OLLAMA_MODEL,
                    label="Ollama Model",
                    info="Model identifier (e.g., llama3.2:latest)",
                    interactive=True,
                )
                ollama_url = gr.Textbox(
                    value=DEFAULT_OLLAMA_BASE_URL,
                    label="Ollama Base URL",
                    info="Full URL including protocol and port",
                    interactive=True,
                )
                rag_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_RAG_TEMPERATURE,
                    step=0.05,
                    label="Temperature",
                    info="Lower = more deterministic, Higher = more creative",
                    interactive=True,
                )
                rag_max_tokens = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    value=DEFAULT_RAG_MAX_TOKENS,
                    step=64,
                    label="Max Tokens",
                    info="Maximum length of generated response",
                    interactive=True,
                )

            gr.Markdown("### Response Settings")
            with gr.Group():
                max_cites = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=DEFAULT_MAX_CITES,
                    step=1,
                    label="Max Citations in Response",
                    info="Maximum source citations in the RAG answer.",
                    interactive=True,
                )
                max_context_doc_length = gr.Slider(
                    minimum=100,
                    maximum=5000,
                    value=DEFAULT_MAX_CONTEXT_DOC_LENGTH,
                    step=50,
                    label="Max Document Context Length",
                    info="Maximum characters per document injected into the RAG prompt (max_doc_content_length).",
                    interactive=True,
                )
                rag_template = gr.Radio(
                    choices=["Basic", "Domain Specific", "Chain Of Thougth"],
                    value=_to_title_case(DEFAULT_RAG_TEMPLATE),
                    label="RAG Template",
                    info="Prompt template to use in RAG module",
                )

            gr.Markdown("### Connection Test")
            with gr.Group():
                test_connection_btn = gr.Button(
                    "🔗 Test Ollama Connection", variant="secondary"
                )
                connection_status = gr.Textbox(
                    label="Connection Result", interactive=False, lines=3
                )

            def test_ollama_connection(model, url):
                try:
                    import requests

                    response = requests.get(f"{url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m.get("name") for m in models]
                        if model in model_names:
                            return f"[OK] Connected to {url}\n[OK] Model '{model}' available\n[OK] Service healthy"
                        return f"[OK] Connected but model '{model}' not found.\nAvailable: {', '.join(model_names[:3])}"
                    return f"[X] Connection failed: HTTP {response.status_code}"
                except requests.exceptions.ConnectionError:
                    return f"[X] Cannot connect to {url}. Is Ollama running?"
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            test_connection_btn.click(
                test_ollama_connection,
                inputs=[ollama_model, ollama_url],
                outputs=connection_status,
            )

            with gr.Group():
                save_rag_btn = gr.Button("💾 Save RAG Settings", variant="primary")
                rag_status = gr.Textbox(label="Status", interactive=False, lines=1)

            def save_rag_settings(
                ollama_model: str,
                ollama_url: str,
                rag_template_title: str,
                rag_temperature: int,
                rag_max_tokens: int,
                max_cites: int,
                max_context_doc_length: int,
                session_state: UIState,
            ) -> tuple[str, UIState]:
                """Persist query settings into session state for use by backend."""
                session_state.settings["ollama_model"] = ollama_model
                session_state.settings["ollama_base_url"] = ollama_url
                session_state.settings["rag_template"] = _to_snake_case(rag_template_title)
                session_state.settings["rag_temperature"] = rag_temperature
                session_state.settings["rag_max_tokens"] = rag_max_tokens
                session_state.settings["max_cites"] = max_cites
                session_state.settings["max_context_doc_length"] = max_context_doc_length
                orchestrator_service.sync_backend(session_state.settings)
                
                status = (
                    f"[OK] Settings saved — \n"
                    f"Model={ollama_model}, URL={ollama_url}, \n"
                    f"Template={rag_template_title}, \n"
                    f"Temperature={rag_temperature}, MaxTokens={rag_max_tokens}, \n"
                    f"MaxCites={max_cites}, MaxContextDocLength={max_context_doc_length}"
                )
                return status, session_state

            save_rag_btn.click(
                save_rag_settings,
                inputs=[
                    ollama_model,
                    ollama_url,
                    rag_template,
                    rag_temperature,
                    rag_max_tokens,
                    max_cites,
                    max_context_doc_length,
                    app_state,
                ],
                outputs=[rag_status, app_state],
            )

        with gr.TabItem("Crawler Configuration"):
            gr.Markdown("### Data Acquisition Settings")
            with gr.Group():
                max_articles_per_spider = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    value=DEFAULT_MAX_ARTICLES_PER_SPIDER,
                    step=100,
                    label="Max Articles per Spider",
                    info="Maximum documents per crawler",
                    interactive=True,
                )
                force_recrawl = gr.Checkbox(
                    value=DEFAULT_FORCE_RECRAWL,
                    label="Force Recrawl",
                    info="Ignore cached data and re-execute all crawlers",
                    interactive=True,
                )
                crawler_status = gr.Textbox(label="Status", interactive=False, lines=1)
                save_crawler_btn = gr.Button("💾 Save Crawlers Settings", variant="primary")

            def save_crawler_settings(
                max_articles_per_spider: int,
                force_recrawl: bool,
                session_state: UIState,
            ) -> tuple[str, UIState]:
                """Persist query settings into session state for use by backend."""
                session_state.settings["max_articles_per_spider"] = max_articles_per_spider
                session_state.settings["force_recrawl"] = force_recrawl
                orchestrator_service.sync_backend(session_state.settings)
                
                status = (
                    f"[OK] Settings saved — \n"
                    f"MaxArticlesPerSpider={max_articles_per_spider}, \n"
                    f"ForceRecrawl={'on' if force_recrawl else 'off'}"
                )
                return status, session_state

            save_crawler_btn.click(
                save_crawler_settings,
                inputs=[
                    max_articles_per_spider,
                    force_recrawl,
                    app_state
                ],
                outputs=[crawler_status, app_state],
            )

        with gr.TabItem("Database Management"):
            gr.Markdown("### Database Operations")
            with gr.Group():
                    clear_raw = gr.Checkbox(
                        value=DEFAULT_CLEAR_RAW,
                        label="Clear Raw",
                        info="Clear cached raw data",
                        interactive=True,
                    )
                    auto_reload_in_db = gr.Checkbox(
                        value=DEFAULT_AUTO_RELOAD,
                        label="Auto-reload Empty Database",
                        info="Automatically execute crawlers if index is below minimum",
                    )
                    use_initial_corpus = gr.Checkbox(
                        value=DEFAULT_USE_INITIAL_CORPUS,
                        label="Use Initial Corpus",
                        info="Load data from initial corpus",
                    )

            with gr.Group():
                gr.Markdown("#### Quick Actions")
                row1, row2, row3 = gr.Row(), gr.Row(), gr.Row()
                with row1:
                    clear_db_btn = gr.Button(
                        "🗑️  Clear All Data", variant="stop", scale=1
                    )
                with row2:
                    reindex_db_btn = gr.Button(
                        "♻️  Reindex Database", variant="secondary", scale=1
                    )
                with row3:
                    reload_db_btn = gr.Button(
                        "📥 Reload from Crawlers", variant="secondary", scale=1
                    )
            operation_output = gr.Textbox(
                label="Operation Result", interactive=False, lines=4
            )
            gr.Markdown("#### Database Status")
            db_info = gr.Textbox(
                label="Current Database Info", interactive=False, lines=5
            )
            with gr.Group():
                refresh_status_btn = gr.Button("🔄 Refresh Status")

            def clear_database(clear_raw: bool):
                try:
                    result = orchestrator_service._get_orchestrator().clear_all_indices(clear_raw=clear_raw)
                    
                    if result.get("success"):
                        return f"[OK] {result.get('message', 'Database cleared successfully')}"
                    return f"[X] {result.get('message', 'Clear operation failed')}"
                
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            def reindex_database(auto_reload: bool, use_initial_corpus: bool):
                try:
                    result = orchestrator_service._get_orchestrator().reindex_database(
                                                                                        auto_reload=auto_reload, 
                                                                                        use_initial_corpus=use_initial_corpus
                                                                                        )
                    if result.get("success"):
                        return (f"[OK] {result.get('message', 'Reindex completed')}\n"
                                f"Indexed: {result.get('indexed_documents', 0)}\n"
                                f"Duration: {result.get('duration_seconds', 0):.2f}s")
                    return (
                        f"[X] {result.get('message', 'Reindex failed')}"
                    )
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            def reload_from_crawlers(use_initial_corpus: bool):
                try:
                    result = orchestrator_service._get_orchestrator().load_documents_from_crawlers(use_initial_corpus=use_initial_corpus)

                    if result.get("success"):
                        return f"[OK] Crawlers executed\nTotal: {result.get('total_documents', 0)}\nIndexed: {result.get('indexed_documents', 0)}\nDuration: {result.get('duration_seconds', 0):.2f}s"
                    return f"[X] {result.get('message', 'Crawler execution failed: Unknown error')}"
                except Exception as e:
                    return f"[X] Error: {str(e)}"

            def get_database_info():
                try:
                    health = (
                        orchestrator_service._get_orchestrator().check_database_health()
                    )
                    return f"Indexed documents: {health.get('document_count', 0)}\nFile documents: {health.get('file_document_count', 0)}\nStatus: {health.get('status', 'unknown')}\nChromaDB: {'[OK] Available' if health.get('has_chromadb') else '[X] Not available'}"
                except Exception as e:
                    return f"[X] Error retrieving status: {str(e)}"

            clear_db_btn.click(
                clear_database,
                inputs=[clear_raw],
                outputs=operation_output
                )
            reindex_db_btn.click(
                reindex_database,
                inputs=[auto_reload_in_db, use_initial_corpus],
                outputs=operation_output
                )
            reload_db_btn.click(
                reload_from_crawlers, 
                inputs=[use_initial_corpus],
                outputs=operation_output)
            refresh_status_btn.click(get_database_info, outputs=db_info)
