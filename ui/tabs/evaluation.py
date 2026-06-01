"""Evaluation Tab - Test execution and metrics visualization."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from ..components.charts import (
    render_evaluation_results_html,
    try_plotly_metrics_chart,
)
from ..services.orchestrator_service import get_orchestrator_service


def build_evaluation_tab() -> None:
    """
    Build evaluation interface with test designer and metrics visualization.

    Supports both default test loading and custom test design.
    """
    orchestrator_service = get_orchestrator_service()

    gr.Markdown("## Evaluation System")
    gr.Markdown(
        "Design custom tests or run default benchmarks. "
        "Evaluate retrieval performance with standard IR metrics."
    )

    test_queries_state = gr.State(value=[])
    evaluation_result_state = gr.State(value={})

    # Placeholders so buttons can target these outputs from other tabs
    test_queries_display = gr.Dataframe(
        headers=["Query ID", "Query Text", "# Relevant", "Grade Scale"],
        label="Current Test Queries",
        interactive=False,
        type="pandas",
    )
    results_display = gr.HTML(label="Results")

    with gr.Tabs():
        # ===== Test Configuration Tab =====
        with gr.TabItem("Test Configuration"):
            gr.Markdown("### Select Test Mode")

            gr.Radio(
                choices=["Load Default Test", "Design Custom Test"],
                value="Load Default Test",
                label="Test Mode",
                info="Load existing test file or create custom tests",
            )

            status_message = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Group():
                gr.Markdown("### Quick Load")
                load_default_btn = gr.Button("📂 Load Default Test", variant="primary")

                def load_default_tests() -> tuple[str, list, list, str]:
                    """Load test queries from JSON file and return table data + simple HTML listing."""
                    test_file = Path("data/test_queries.json")
                    if not test_file.exists():
                        return (
                            "[X] test_queries.json not found",
                            [],
                            [],
                            "<p>test_queries.json not found</p>",
                        )

                    try:
                        with open(test_file, encoding="utf-8") as f:
                            data = json.load(f)
                            tests = data.get("test_queries", [])
                            count = len(tests)

                            # Build a simple table representation for the dataframe
                            table_data = [
                                [
                                    t.get("query_id", ""),
                                    (t.get("query", "")[:40] + "...")
                                    if len(t.get("query", "")) > 40
                                    else t.get("query", ""),
                                    len(t.get("relevant", [])),
                                    "Graded" if t.get("grades") else "Binary",
                                ]
                                for t in tests
                            ]

                            # Create a simple HTML listing to show loaded queries in results panel
                            results_html = "<div><h4>Loaded Test Queries</h4><ul>"
                            for t in tests:
                                qid = t.get("query_id", "-")
                                qtxt = t.get("query", "")
                                results_html += (
                                    f"<li><strong>{qid}</strong>: {qtxt}</li>"
                                )
                            results_html += "</ul></div>"

                            return (
                                f"[OK] Loaded {count} test queries",
                                tests,
                                table_data,
                                results_html,
                            )
                    except Exception as e:
                        return (
                            f"[X] Error loading file: {str(e)}",
                            [],
                            [],
                            "<p>Error loading tests</p>",
                        )

                load_default_btn.click(
                    load_default_tests,
                    outputs=[
                        status_message,
                        test_queries_state,
                        test_queries_display,
                        results_display,
                    ],
                )

        # ===== Test Designer Tab =====
        with gr.TabItem("Test Designer"):
            gr.Markdown("### Create Custom Test Queries")
            gr.Markdown(
                "Add queries and mark relevant documents. "
                "Optionally assign relevance grades (0-3 scale)."
            )

            with gr.Group():
                gr.Markdown("#### Add Test Query")
                query_id = gr.Textbox(placeholder="q1", label="Query ID")
                query_text = gr.Textbox(
                    placeholder="Enter search query", label="Query Text"
                )

                grade_type = gr.Radio(
                    choices=["Binary (Relevant/Not)", "Graded (0-3)"],
                    value="Binary (Relevant/Not)",
                    label="Relevance Scale",
                )

                relevant_docs = gr.Textbox(
                    placeholder="doc_id1, doc_id2, doc_id3",
                    label="Relevant Document IDs (comma-separated)",
                    info="IDs of documents relevant to this query",
                )

                grades_input = gr.Textbox(
                    placeholder="doc_id1:3, doc_id2:2",
                    label="Grades (if graded scale selected)",
                    info="Format: doc_id:grade, doc_id:grade (0=not relevant, 3=highly relevant)",
                )

                add_query_btn = gr.Button("➕ Add Query to Set", variant="secondary")

            test_set_message = gr.Textbox(
                label="Test Set Status", interactive=False, lines=2
            )

            def add_query_to_set(
                q_id: str, q_text: str, g_type: str, rel_docs: str, grades: str
            ) -> tuple[str, list, list]:
                """Add a single query to test set."""
                if not q_id or not q_text or not rel_docs:
                    return (
                        "[X] Query ID, text, and relevant docs are required",
                        test_queries_state.value,
                        [],
                    )

                try:
                    doc_list = [d.strip() for d in rel_docs.split(",")]

                    query_entry = {
                        "query_id": q_id.strip(),
                        "query": q_text.strip(),
                        "relevant": doc_list,
                    }

                    if g_type == "Graded (0-3)" and grades:
                        grade_dict = {}
                        for pair in grades.split(","):
                            doc_id, grade = pair.strip().split(":")
                            grade_dict[doc_id.strip()] = int(grade.strip())
                        query_entry["grades"] = grade_dict

                    current = test_queries_state.value or []
                    current.append(query_entry)

                    table_data = [
                        [
                            q["query_id"],
                            (q["query"][:40] + "...")
                            if len(q["query"]) > 40
                            else q["query"],
                            len(q.get("relevant", [])),
                            "Graded" if "grades" in q else "Binary",
                        ]
                        for q in current
                    ]

                    return (
                        f"[OK] Added query '{q_id}' to test set ({len(current)} total)",
                        current,
                        table_data,
                    )
                except Exception as e:
                    return f"[X] Error: {str(e)}", test_queries_state.value, []

            add_query_btn.click(
                add_query_to_set,
                inputs=[query_id, query_text, grade_type, relevant_docs, grades_input],
                outputs=[test_set_message, test_queries_state, test_queries_display],
            )

        # ===== Evaluation Execution Tab =====
        with gr.TabItem("Run Evaluation"):
            gr.Markdown("### Execute Evaluation")

            with gr.Group():
                execution_info = gr.Textbox(
                    label="Test Set Info", interactive=False, lines=2
                )

                run_eval_btn = gr.Button(
                    "▶️  Run Evaluation", variant="primary", size="lg"
                )

            eval_progress = gr.Textbox(label="Progress", interactive=False, lines=2)

            def get_test_info() -> str:
                """Get current test set information."""
                tests = test_queries_state.value
                if not tests:
                    return "No test set loaded"
                return f"Test queries: {len(tests)}\nStatus: Ready to run"

            def run_evaluation() -> tuple[str]:
                """Execute evaluation on test set."""
                tests = test_queries_state.value
                if not tests:
                    return ("[X] No test set loaded",)

                try:
                    test_spec = {"test_queries": tests}
                    result = orchestrator_service.evaluate_test(test_spec)

                    if result.get("status") == "success":
                        eval_progress_msg = (
                            f"[OK] Evaluation completed\n"
                            f"Queries evaluated: {result.get('aggregate', {}).get('num_queries', 0)}\n"
                            f"Time: {result.get('execution_time_seconds', 0):.2f}s"
                        )
                        evaluation_result_state.value = result
                        return (eval_progress_msg,)
                    else:
                        return (
                            f"[X] Evaluation failed:\n{result.get('message', 'Unknown error')}",
                        )
                except Exception as e:
                    return (f"[X] Error: {str(e)}",)

            execution_info.value = get_test_info()
            run_eval_btn.click(run_evaluation, outputs=[eval_progress])

        # ===== Results Visualization Tab =====
        with gr.TabItem("Results"):
            gr.Markdown("### Evaluation Results")

            def display_results() -> str:
                """Display evaluation results with visualization."""
                result = evaluation_result_state.value
                if not result or result.get("status") != "success":
                    return (
                        "<p>No evaluation results available. Run evaluation first.</p>"
                    )

                aggregate = result.get("aggregate", {})
                per_query = result.get("per_query", [])
                exec_time = result.get("execution_time_seconds", 0)

                plotly_html = try_plotly_metrics_chart(result)
                if plotly_html:
                    html_content = f"<h3>Metrics Visualization</h3>\n{plotly_html}"
                else:
                    html_content = ""

                html_content += "\n" + render_evaluation_results_html(
                    aggregate, per_query, exec_time
                )
                return html_content

            display_results_btn = gr.Button("📊 Display Results")
            display_results_btn.click(display_results, outputs=[results_display])
