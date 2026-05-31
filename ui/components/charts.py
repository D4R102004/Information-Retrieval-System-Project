"""Chart utilities with Plotly support and HTML fallback."""

from typing import Dict, List, Any, Optional


def create_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Create an HTML table from evaluation metrics.

    Args:
        metrics: Dictionary of metric name -> value

    Returns:
        HTML table string
    """
    if not metrics:
        return "<p>No metrics available</p>"

    rows = "".join(
        f"<tr><td><strong>{key}</strong></td><td>{value:.4f}</td></tr>"
        for key, value in metrics.items()
    )

    return f"""
    <table style="border-collapse: collapse; width: 100%;">
        <thead>
            <tr style="background-color: #f0f0f0;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def create_per_query_table(per_query_results: List[Dict[str, Any]]) -> str:
    """
    Create an HTML table for per-query evaluation results.

    Args:
        per_query_results: List of per-query result dictionaries

    Returns:
        HTML table string
    """
    if not per_query_results:
        return "<p>No per-query results available</p>"

    columns = list(per_query_results[0].keys())

    header_row = "".join(
        f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{col}</th>"
        for col in columns
    )

    data_rows = ""
    for result in per_query_results:
        cells = "".join(
            f"<td style='border: 1px solid #ddd; padding: 8px;'>{str(result.get(col, 'N/A'))}</td>"
            for col in columns
        )
        data_rows += f"<tr>{cells}</tr>"

    return f"""
    <table style="border-collapse: collapse; width: 100%; font-size: 0.9em;">
        <thead>
            <tr style="background-color: #f0f0f0;">
                {header_row}
            </tr>
        </thead>
        <tbody>
            {data_rows}
        </tbody>
    </table>
    """


def create_status_card(title: str, content: str, status: str = "neutral") -> str:
    """
    Create a styled status card.

    Args:
        title: Card title
        content: Card content
        status: Status type ('healthy', 'warning', 'error', 'neutral')

    Returns:
        HTML card string
    """
    color_map = {
        "healthy": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "neutral": "#6B7280"
    }
    color = color_map.get(status, color_map["neutral"])

    return f"""
    <div style="
        border: 2px solid {color};
        border-left: 5px solid {color};
        padding: 16px;
        margin: 8px 0;
        border-radius: 4px;
        background-color: transparent;
    ">
        <div style="font-weight: bold; margin-bottom: 8px;">{title}</div>
        <div style="color: inherit; font-size: 0.95em;">{content}</div>
    </div>
    """


def render_evaluation_results_html(
    aggregate: Dict[str, Any],
    per_query: List[Dict[str, Any]],
    execution_time: float
) -> str:
    """
    Render complete evaluation results as HTML.

    Args:
        aggregate: Aggregate metrics dictionary
        per_query: List of per-query results
        execution_time: Total execution time in seconds

    Returns:
        HTML string with full results
    """
    aggregate_table = create_metrics_table(aggregate) if aggregate else "<p>No aggregate metrics</p>"
    per_query_table = create_per_query_table(per_query) if per_query else "<p>No per-query results</p>"

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <h3>Aggregate Metrics</h3>
        {aggregate_table}
        
        <h3 style="margin-top: 24px;">Per-Query Results</h3>
        {per_query_table}
        
        <p style="color: #6B7280; font-size: 0.9em; margin-top: 16px;">
            Execution time: {execution_time:.2f}s
        </p>
    </div>
    """


def try_plotly_metrics_chart(evaluation_result: Dict[str, Any]) -> Optional[str]:
    """
    Try to create a Plotly metrics visualization.

    Returns None if plotly is unavailable, allowing fallback to HTML.

    Args:
        evaluation_result: Evaluation result dictionary with 'aggregate' key

    Returns:
        HTML string with embedded Plotly chart, or None if unavailable
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        aggregate = evaluation_result.get("aggregate", {})
        if not aggregate:
            return None

        metric_names = list(aggregate.keys())
        metric_values = list(aggregate.values())

        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker=dict(color="#3B82F6"),
                text=[f"{v:.3f}" for v in metric_values],
                textposition="outside"
            )
        ])

        fig.update_layout(
            title="Evaluation Metrics",
            xaxis_title="Metric",
            yaxis_title="Score",
            height=400,
            margin=dict(b=100),
            template="plotly_white"
        )

        return pio.to_html(fig, include_plotlyjs="cdn", div_id="metrics_chart")

    except ImportError:
        return None
    except Exception:
        return None
