from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def build_charts(equity_curve: pd.DataFrame, title: str) -> dict[str, go.Figure]:
    curve = equity_curve.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])

    equity_fig = go.Figure(
        go.Scatter(x=curve["timestamp"], y=curve["equity"], mode="lines", name="Equity")
    )
    equity_fig.update_layout(
        title=f"{title}: equity curve", xaxis_title="Session", yaxis_title="EUR"
    )

    drawdown = curve["equity"] / curve["equity"].cummax() - 1
    drawdown_fig = go.Figure(
        go.Scatter(x=curve["timestamp"], y=drawdown, mode="lines", fill="tozeroy", name="Drawdown")
    )
    drawdown_fig.update_layout(
        title=f"{title}: drawdown", xaxis_title="Session", yaxis_title="Drawdown"
    )

    daily_returns = curve.set_index("timestamp")["equity"].pct_change().dropna()
    monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
    monthly_fig = go.Figure(
        go.Bar(x=monthly_returns.index, y=monthly_returns, name="Monthly return")
    )
    monthly_fig.update_layout(
        title=f"{title}: monthly returns", xaxis_title="Month", yaxis_title="Return"
    )
    return {
        "equity_curve": equity_fig,
        "drawdown": drawdown_fig,
        "monthly_returns": monthly_fig,
    }


def write_charts(equity_curve: pd.DataFrame, output_dir: str | Path, title: str) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, figure in build_charts(equity_curve, title).items():
        path = root / f"{name}.html"
        figure.write_html(path, include_plotlyjs="directory")
        written.append(path)
    return written
