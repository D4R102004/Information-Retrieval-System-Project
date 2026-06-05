"""System Status Tab - Diagnostics and monitoring."""

from __future__ import annotations

import gradio as gr
from typing import Any

from ..services.orchestrator_service import get_orchestrator_service
from ..components.charts import create_status_card


def build_status_tab() -> None:
    """
    Build system status and diagnostics interface.

    Displays health checks, database state, crawler status, and LLM connectivity.
    """
    orchestrator_service = get_orchestrator_service()

    gr.Markdown("## System Status")
    gr.Markdown("Real-time diagnostics of system health, database, and services.")

    with gr.Group():
        refresh_btn = gr.Button("🔄 Refresh All Status", variant="primary")
        last_update = gr.Textbox(
            label="Last Updated",
            interactive=False,
            scale=1
        )

    with gr.Tabs():
        # ===== Health Summary Tab =====
        with gr.TabItem("Health Summary"):
            gr.Markdown("### Overall System Status")
            health_summary = gr.HTML(label="Health Overview")

        # ===== Database Status Tab =====
        with gr.TabItem("Database"):
            gr.Markdown("### Database Statistics")
            db_status = gr.HTML(label="Database Status")

        # ===== Crawler Status Tab =====
        with gr.TabItem("Crawlers"):
            gr.Markdown("### Data Acquisition Status")
            crawler_status = gr.HTML(label="Crawler Status")

        # ===== LLM Service Tab =====
        with gr.TabItem("LLM Service"):
            gr.Markdown("### Large Language Model Connectivity")
            llm_status = gr.HTML(label="LLM Status")

    def get_health_summary() -> str:
        """Generate health summary display."""
        try:
            health = orchestrator_service.check_database_health()
            status = health.get("status", "unknown")
            doc_count = health.get("document_count", 0)
            is_empty = health.get("is_empty", True)

            if is_empty:
                status_type = "error"
                message = "Database is empty. No indexed documents available."
            elif doc_count < 500:
                status_type = "warning"
                message = f"Low document count: {doc_count} (minimum 500 recommended)"
            else:
                status_type = "healthy"
                message = f"Database healthy with {doc_count} documents"

            status_icon = {
                "error": "🔴 Critical",
                "warning": "🟡 Degraded",
                "healthy": "🟢 Healthy"
            }.get(status_type, "⚪ Unknown")

            return create_status_card(
                title=f"Overall Status: {status_icon}",
                content=message,
                status=status_type
            )
        except Exception as e:
            return create_status_card(
                title="Status Check Failed",
                content=f"Error: {str(e)}",
                status="error"
            )

    def get_database_status() -> str:
        """Generate database status display."""
        try:
            health = orchestrator_service.check_database_health()
            indexed = health.get("document_count", 0)
            file_docs = health.get("file_document_count", 0)
            has_chroma = health.get("has_chromadb", False)

            raw_count = 0
            try:
                raw_count = orchestrator_service.count_raw_documents()
            except Exception:
                raw_count = 0

            db_info = f"""
            <div style="font-family: monospace; padding: 12px; border-radius: 4px; border-left: 4px solid #3B82F6;">
                <div>📊 <strong>Indexed Documents:</strong> {indexed}</div>
                <div>📄 <strong>File Documents:</strong> {file_docs}</div>
                <div>🗂️  <strong>Raw Documents:</strong> {raw_count}</div>
                <div>🗄️  <strong>ChromaDB:</strong> {'[OK] Available' if has_chroma else '[X] Not available'}</div>
                <div style="margin-top: 8px; font-size: 0.9em; opacity: 0.8;">
                    Status: {health.get('status', 'unknown').upper()}
                </div>
            </div>
            """
            return db_info
        except Exception as e:
            return f"<p style='color: #EF4444;'>Error retrieving database status: {str(e)}</p>"

    def get_crawler_status() -> str:
        """Generate crawler status display."""
        try:
            crawlers = ["DevTo", "HackerNews", "RealPython", "Lobsters", "TheNewStack", "TheVerge"]
            
            crawler_html = "<div style='font-family: monospace;'>"
            for crawler in crawlers:
                doc_count = orchestrator_service.count_raw_documents(crawler)
                last_crawled = orchestrator_service.get_last_crawled_date(crawler)
                crawler_html += f"""
                <div style="padding: 12px; margin: 8px 0; border-radius: 4px; border-left: 3px solid #3B82F6;">
                    <div><strong>{crawler}</strong></div>
                    <div style="font-size: 0.9em; opacity: 0.8;">Last used: {last_crawled or "Never"} | Documents: {doc_count or "0"}</div>
                </div>
                """
            crawler_html += "</div>"
            return crawler_html
        except Exception as e:
            return f"<p style='color: #EF4444;'>Error retrieving crawler status: {str(e)}</p>"

    def get_llm_status() -> str:
        """Generate LLM service status display."""
        try:
            base_url = "http://localhost:11434"
            model = "llama3.2:latest"

            try:
                import requests
                response = requests.get(f"{base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name") for m in models]
                    
                    model_available = model in model_names
                    status_color = "#10B981" if model_available else "#F59E0B"
                    status_indicator = "[OK]" if model_available else "⚠️"

                    llm_html = f"""
                    <div style="font-family: monospace; padding: 12px; border-radius: 4px; border-left: 4px solid {status_color};">
                        <div>🔗 <strong>Status:</strong> <span style="color: {status_color};">{status_indicator} Connected</span></div>
                        <div>🏥 <strong>URL:</strong> {base_url}</div>
                        <div>🤖 <strong>Active Model:</strong> {model} {' [OK]' if model_available else ' (not loaded)'}</div>
                        <div style="font-size: 0.9em; margin-top: 8px; opacity: 0.8;">
                            Available models: {', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''}
                        </div>
                    </div>
                    """
                    return llm_html
                else:
                    return create_status_card(
                        title="Connection Error",
                        content=f"Ollama returned HTTP {response.status_code}",
                        status="error"
                    )
            except requests.exceptions.ConnectionError:
                return create_status_card(
                    title="Not Connected",
                    content=f"Cannot reach {base_url}. Is Ollama running?",
                    status="error"
                )
        except Exception as e:
            return create_status_card(
                title="Status Check Failed",
                content=f"Error: {str(e)}",
                status="error"
            )

    def get_timestamp() -> str:
        """Get current timestamp."""
        from datetime import datetime, timezone
        return f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"

    def refresh_all_status() -> tuple[str, str, str, str, str]:
        """Refresh all status displays."""
        health_html = get_health_summary()
        db_html = get_database_status()
        crawler_html = get_crawler_status()
        llm_html = get_llm_status()
        timestamp = get_timestamp()

        return health_html, db_html, crawler_html, llm_html, timestamp

    refresh_btn.click(
        refresh_all_status,
        outputs=[health_summary, db_status, crawler_status, llm_status, last_update]
    )
