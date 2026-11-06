from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
from plotly.utils import PlotlyJSONEncoder

from quant_trader.reporting.charts import build_charts


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _markdown_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id, "language": "markdown"},
        "source": _source_lines(source),
    }


def _code_cell(
    source: str, cell_id: str, outputs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id, "language": "python"},
        "outputs": outputs or [],
        "source": _source_lines(source),
    }


def _chart_output(figure: Any) -> dict[str, Any]:
    specification = json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
    return {
        "output_type": "display_data",
        "data": {
            "application/vnd.plotly.v1+json": {
                "data": specification["data"],
                "layout": specification["layout"],
            },
            "text/plain": [f"Plotly figure with {len(figure.data)} traces"],
        },
        "metadata": {},
    }


def _chart_source(chart_name: str, title: str) -> str:
    title_literal = repr(title)
    common = (
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n\n"
        'equity_curve = pd.read_csv("equity_curve.csv", parse_dates=["timestamp"])\n'
    )
    if chart_name == "equity_curve":
        body = (
            "figure = go.Figure(go.Scatter(\n"
            '    x=equity_curve["timestamp"], y=equity_curve["equity"],\n'
            '    mode="lines", name="Equity"\n'
            "))\n"
            f"figure.update_layout(title={title_literal}, xaxis_title='Session', yaxis_title='EUR')\n"
        )
    elif chart_name == "drawdown":
        body = (
            'drawdown = equity_curve["equity"] / equity_curve["equity"].cummax() - 1\n'
            "figure = go.Figure(go.Scatter(\n"
            '    x=equity_curve["timestamp"], y=drawdown, mode="lines",\n'
            '    fill="tozeroy", name="Drawdown"\n'
            "))\n"
            f"figure.update_layout(title={title_literal}, xaxis_title='Session', yaxis_title='Drawdown')\n"
        )
    else:
        body = (
            'daily_returns = equity_curve.set_index("timestamp")["equity"].pct_change().dropna()\n'
            'monthly_returns = (1 + daily_returns).resample("ME").prod() - 1\n'
            'figure = go.Figure(go.Bar(x=monthly_returns.index, y=monthly_returns, name="Monthly return"))\n'
            f"figure.update_layout(title={title_literal}, xaxis_title='Month', yaxis_title='Return')\n"
        )
    return common + body + "figure.show()\n"


def _display_table(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "output_type": "display_data",
        "data": {
            "text/html": [frame.to_html(index=False, border=0, max_rows=None, max_cols=None)],
            "text/plain": [frame.to_string(index=False, max_rows=20)],
        },
        "metadata": {},
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_report_notebook(report: dict[str, Any], output_dir: str | Path) -> Path:
    root = Path(output_dir)
    metadata = report.get("metadata", {})
    metrics = report.get("metrics", {})
    strategy = str(metadata.get("strategy", "Backtest"))
    period = f"{metadata.get('start_date', 'N/A')} to {metadata.get('end_date', 'N/A')}"
    summary = [
        f"# Backtest report: {strategy}",
        "",
        f"- Period: {period}",
        f"- Data source: {metadata.get('data_source', 'N/A')}",
        (
            f"- Coverage: {metadata.get('data_coverage', {}).get('start', 'N/A')} to "
            f"{metadata.get('data_coverage', {}).get('end', 'N/A')}"
        ),
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in metrics.items():
        if value is None:
            display = "N/A"
        elif isinstance(value, (int, float)):
            display = f"{value:,.4f}"
        else:
            display = str(value)
        summary.append(f"| {name.replace('_', ' ').title()} | {display} |")

    cells: list[dict[str, Any]] = [_markdown_cell("\n".join(summary), "report-summary")]
    cells.append(
        _markdown_cell(
            "## Charts\n\nInteractive Plotly charts are embedded below.", "report-charts"
        )
    )
    chart_title = strategy
    for index, (name, figure) in enumerate(
        build_charts(pd.read_csv(root / "equity_curve.csv"), chart_title).items()
    ):
        cells.append(_markdown_cell(f"### {figure.layout.title.text}", f"chart-title-{index}"))
        cells.append(
            _code_cell(
                _chart_source(name, str(figure.layout.title.text)),
                f"chart-{name}",
                [_chart_output(figure)],
            )
        )

    cells.append(
        _markdown_cell("## CSV Data\n\nThe full report CSVs are displayed below.", "report-csvs")
    )
    for index, filename in enumerate(("equity_curve.csv", "trades.csv", "transactions.csv")):
        path = root / filename
        frame = _read_csv(path)
        cells.append(_markdown_cell(f"### {filename}", f"csv-title-{index}"))
        source = (
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            f"csv_path = Path({filename!r})\n"
            "try:\n"
            "    frame = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()\n"
            "except pd.errors.EmptyDataError:\n"
            "    frame = pd.DataFrame()\n"
            "display(frame)\n"
        )
        cells.append(_code_cell(source, f"csv-{index}", [_display_table(frame)]))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = root / "report.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
