from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from pytrading import Portfolio
from pytrading.experiment_runner import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START_DATE,
    _git_revision,
    _json_safe_records,
    build_experiment_matrix,
    default_broker_config,
)
from pytrading.strategies.momentum import run_experiment_cases

ROLLING_WINDOW_YEARS = 3
ROLLING_STEP_YEARS = 1
VARIANT_PAIRS = (
    ("AB-M5", "ABC-M5"),
    ("AB-M10", "ABC-M10"),
    ("AB-M5", "AB-M10"),
    ("ABC-M5", "ABC-M10"),
)
METRIC_ALIASES = {
    "Sharpe Ratio": "Sharpe",
    "Sortino Ratio": "Sortino",
    "Calmar Ratio": "Calmar",
    "Average Cash Allocation": "Cash Allocation",
}
ROLLING_METRICS = (
    "Net End Value",
    "Gross End Value",
    "Gross-Net End Value Difference",
    "Final Portfolio Value",
    "Gross Final Portfolio Value",
    "Total Return",
    "Net Total Return",
    "Net CAGR",
    "Gross Total Return",
    "Gross CAGR",
    "Maximum Drawdown",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Average Realized Volatility",
    "Downside Volatility",
    "Average Covariance-Based Portfolio Volatility",
    "Weight-Weighted Average Constituent Volatility",
    "Average Position Volatility",
    "Largest Loss Month",
    "Positive Months",
    "Negative Months",
    "Positive Month Share",
    "Profit Factor",
    "Number of Trades",
    "Buy Orders",
    "Sell Orders",
    "Completed Trade Lots",
    "Completed Positions",
    "Open Lots",
    "Open Positions",
    "Average Holding Period (sessions)",
    "Median Holding Period (sessions)",
    "Weighted Average Holding Period (sessions)",
    "Average Position Duration (sessions)",
    "Average Open Position Duration (sessions)",
    "Average Exposure",
    "Maximum Exposure",
    "Cash Allocation",
    "Average Cash Allocation",
    "Time with Cash",
    "Average Position Weight",
    "Median Position Weight",
    "Largest Position Weight",
    "Weight Concentration HHI",
    "Time at or Above Position Cap",
    "Average EUROPE Exposure",
    "Maximum EUROPE Exposure",
    "Average USA Exposure",
    "Maximum USA Exposure",
    "Average OTHER Exposure",
    "Maximum OTHER Exposure",
    "Annualized One-sided Turnover",
    "Annualized Two-sided Turnover",
    "Annual Transaction Costs",
    "Proportional Fees Costs",
    "Fixed Fees Costs",
    "Transaction Fees",
    "Slippage Costs",
    "Transaction Costs",
    "Cost Share of Gross Profit",
    "Proportional Fees Costs / Initial Capital",
    "Fixed Fees Costs / Initial Capital",
    "Transaction Fees / Initial Capital",
    "Slippage Costs / Initial Capital",
    "Transaction Costs / Initial Capital",
)
NULL_METRIC_REASONS = {
    "Sharpe Ratio": "Daily return standard deviation is zero or unavailable.",
    "Sortino Ratio": "Downside deviation is zero or unavailable.",
    "Calmar Ratio": "Maximum drawdown is zero or unavailable.",
    "Average Realized Volatility": "A valid daily return sample is unavailable.",
    "Downside Volatility": "A valid downside return sample is unavailable.",
    "Average Covariance-Based Portfolio Volatility": (
        "No held-position date has the minimum jointly observed return history."
    ),
    "Weight-Weighted Average Constituent Volatility": (
        "No active position has valid constituent volatility for the window."
    ),
    "Average Position Volatility": (
        "No active position has valid constituent volatility for the window."
    ),
    "Average Holding Period (sessions)": "No FIFO buy lot was fully closed.",
    "Median Holding Period (sessions)": "No FIFO buy lot was fully closed.",
    "Weighted Average Holding Period (sessions)": ("No FIFO buy lot was fully closed."),
    "Average Position Duration (sessions)": (
        "No instrument position episode was fully closed."
    ),
    "Average Open Position Duration (sessions)": (
        "No instrument position remained open at the window end."
    ),
    "Cost Share of Gross Profit": (
        "Gross portfolio profit was non-positive or unavailable."
    ),
    "Profit Factor": "No closed-trade profit/loss sample was available.",
    "Largest Loss Month": "No valid monthly return was available.",
    "Positive Month Share": "No valid monthly return was available.",
}


@dataclass(frozen=True)
class RollingWindow:
    start_date: pd.Timestamp
    end_date_exclusive: pd.Timestamp
    observed_end_exclusive: pd.Timestamp
    complete: bool

    @property
    def window_id(self) -> str:
        return f"{self.start_date:%Y-%m-%d}_{self.end_date_exclusive:%Y-%m-%d}"

    @property
    def backtest_end_date(self) -> str:
        return (self.observed_end_exclusive - pd.Timedelta(days=1)).date().isoformat()


def build_rolling_windows(
    start_date: str | pd.Timestamp,
    latest_data_date: str | pd.Timestamp,
    window_years: int = ROLLING_WINDOW_YEARS,
    step_years: int = ROLLING_STEP_YEARS,
) -> tuple[RollingWindow, ...]:
    start = pd.Timestamp(start_date).normalize()
    latest = pd.Timestamp(latest_data_date).normalize()
    if start > latest:
        raise ValueError("Start date must not be after the latest data date.")
    if window_years <= 0 or step_years <= 0:
        raise ValueError("Window and step lengths must be positive.")

    available_end_exclusive = latest + pd.Timedelta(days=1)
    windows = []
    current = start
    while current <= latest:
        nominal_end = current + pd.DateOffset(years=window_years)
        observed_end = min(nominal_end, available_end_exclusive)
        windows.append(
            RollingWindow(
                start_date=current,
                end_date_exclusive=nominal_end,
                observed_end_exclusive=observed_end,
                complete=nominal_end <= available_end_exclusive,
            )
        )
        current += pd.DateOffset(years=step_years)
    return tuple(windows)


def _monthly_positive_count(equity: pd.Series, initial_capital: float) -> int:
    if equity.empty:
        return 0
    monthly_values = equity.groupby(equity.index.to_period("M")).last()
    monthly_returns = monthly_values.pct_change()
    monthly_returns.iloc[0] = monthly_values.iloc[0] / initial_capital - 1
    return int(monthly_returns.gt(0).sum())


def _null_metric_reasons(row: dict[str, Any]) -> str:
    reasons = {}
    for metric, value in row.items():
        is_rolling_metric = metric in ROLLING_METRICS
        is_benchmark_field = metric == "Benchmarks Available"
        is_annual_field = bool(
            re.fullmatch(
                r"(?:Annualized (?:One-sided|Two-sided) Turnover|"
                r"Annual Transaction Costs) \d{4}",
                metric,
            )
        )
        if not (is_rolling_metric or is_benchmark_field or is_annual_field):
            continue
        if value is not None and not pd.isna(value):
            continue
        if row.get("Status") != "success":
            reasons[metric] = (
                f"Variant failed: {row.get('Error', 'no error detail was recorded')}."
            )
        elif is_benchmark_field:
            reasons[metric] = "No benchmark series were configured or available."
        elif is_annual_field:
            year = int(metric.rsplit(" ", maxsplit=1)[-1])
            start_year = pd.Timestamp(row["Start Date"]).year
            observed_end = row.get("Observed End Date")
            end_year = pd.Timestamp(observed_end).year if observed_end else start_year
            reasons[metric] = (
                "No valuation sessions fell in this calendar year within the window."
                if year < start_year or year > end_year
                else "No annual value was produced; see the metric definition."
            )
        else:
            reasons[metric] = NULL_METRIC_REASONS.get(
                metric,
                "The metric is undefined or unavailable for these observations; "
                "see metric_definitions.md.",
            )
    return json.dumps(reasons, sort_keys=True)


def _enrich_result(
    row: dict[str, Any],
    window: RollingWindow,
    initial_capital: float,
) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if not key.startswith("_")}
    equity = row.get("_Net Equity")
    observed_end = (
        pd.Timestamp(equity.index[-1]).date().isoformat()
        if isinstance(equity, pd.Series) and not equity.empty
        else None
    )
    result.update(
        {
            "Window ID": window.window_id,
            "Start Date": window.start_date.date().isoformat(),
            "End Date": window.end_date_exclusive.date().isoformat(),
            "Observed End Date": observed_end,
            "Window Status": "complete" if window.complete else "incomplete",
            "Complete Window": window.complete,
            "Observed Sessions": len(equity) if isinstance(equity, pd.Series) else 0,
        }
    )
    for alias, source in METRIC_ALIASES.items():
        result[alias] = row.get(source)
    if isinstance(equity, pd.Series) and row.get("Status") == "success":
        result["Positive Months"] = _monthly_positive_count(equity, initial_capital)
    else:
        result["Positive Months"] = None

    for metric in (
        "Proportional Fees Costs",
        "Fixed Fees Costs",
        "Transaction Fees",
        "Slippage Costs",
        "Transaction Costs",
    ):
        value = row.get(metric)
        result[f"{metric} / Initial Capital"] = (
            float(value) / initial_capital
            if value is not None and pd.notna(value)
            else None
        )
    for metric in ROLLING_METRICS:
        result.setdefault(metric, None)
    return result


def aggregate_rolling_results(
    details: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []
    for variant, variant_rows in details.groupby("Experiment ID", sort=False):
        complete_rows = variant_rows.loc[
            variant_rows["Complete Window"].eq(True)
            & variant_rows["Status"].eq("success")
        ]
        returns = pd.to_numeric(
            complete_rows.get("Net Total Return", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        cagr = pd.to_numeric(
            complete_rows.get("Net CAGR", pd.Series(dtype=float)), errors="coerce"
        )
        drawdown = pd.to_numeric(
            complete_rows.get("Maximum Drawdown", pd.Series(dtype=float)),
            errors="coerce",
        )
        sharpe = pd.to_numeric(
            complete_rows.get("Sharpe Ratio", pd.Series(dtype=float)),
            errors="coerce",
        )
        robustness = {
            "Number of Complete Windows": int(
                variant_rows["Complete Window"].eq(True).sum()
            ),
            "Number of Successful Complete Windows": len(complete_rows),
            "Number of Positive Return Windows": int(returns.gt(0).sum()),
            "Number of Negative Return Windows": int(returns.lt(0).sum()),
            "Positive Return Window Share": (
                float(returns.gt(0).mean()) if not returns.empty else None
            ),
            "Worst CAGR": float(cagr.min()) if cagr.notna().any() else None,
            "Best CAGR": float(cagr.max()) if cagr.notna().any() else None,
            "Worst Maximum Drawdown": (
                float(drawdown.min()) if drawdown.notna().any() else None
            ),
            "Worst Sharpe Ratio": (
                float(sharpe.min()) if sharpe.notna().any() else None
            ),
            "Best Sharpe Ratio": (
                float(sharpe.max()) if sharpe.notna().any() else None
            ),
        }
        for metric in ROLLING_METRICS:
            values = pd.to_numeric(
                complete_rows.get(metric, pd.Series(dtype=float)), errors="coerce"
            ).dropna()
            summary_rows.append(
                {
                    "Experiment ID": variant,
                    "Metric": metric,
                    "Minimum": float(values.min()) if not values.empty else None,
                    "Maximum": float(values.max()) if not values.empty else None,
                    "Arithmetic Mean": (
                        float(values.mean()) if not values.empty else None
                    ),
                    "Median": float(values.median()) if not values.empty else None,
                    "Standard Deviation": (
                        float(values.std(ddof=1)) if len(values) > 1 else None
                    ),
                    "Number of Metric Observations": len(values),
                    **robustness,
                }
            )
    return pd.DataFrame(summary_rows)


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _pairwise_comparisons(details: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        ("Net CAGR", "CAGR Difference (pp) (B-A)", 100.0),
        ("Sharpe Ratio", "Sharpe Difference (B-A)", 1.0),
        ("Maximum Drawdown", "Maximum Drawdown Difference (pp) (B-A)", 100.0),
        ("Transaction Costs", "Transaction Costs Difference (EUR) (B-A)", 1.0),
        (
            "Transaction Costs / Initial Capital",
            "Transaction Costs Difference (% capital) (B-A)",
            100.0,
        ),
    )
    by_window_variant = {
        (row["Window ID"], row["Experiment ID"]): row
        for row in details.to_dict(orient="records")
    }
    records: list[dict[str, Any]] = []
    for variant_a, variant_b in VARIANT_PAIRS:
        pair_rows = []
        for window_id in details["Window ID"].drop_duplicates():
            row_a = by_window_variant.get((window_id, variant_a), {})
            row_b = by_window_variant.get((window_id, variant_b), {})
            record: dict[str, Any] = {
                "Record Type": "window",
                "Reference Variant A": variant_a,
                "Compared Variant B": variant_b,
                "Window ID": window_id,
                "Start Date": row_a.get("Start Date") or row_b.get("Start Date"),
                "End Date": row_a.get("End Date") or row_b.get("End Date"),
                "Window Status": row_a.get("Window Status")
                or row_b.get("Window Status"),
                "Complete Window": bool(
                    row_a.get("Complete Window", row_b.get("Complete Window", False))
                ),
            }
            a_success = row_a.get("Status") == "success"
            b_success = row_b.get("Status") == "success"
            for metric, delta_name, scale in metrics:
                value_a = _as_float(row_a.get(metric)) if a_success else None
                value_b = _as_float(row_b.get(metric)) if b_success else None
                difference = (
                    (value_b - value_a) * scale
                    if value_a is not None and value_b is not None
                    else None
                )
                record[delta_name] = difference
                if metric in {"Net CAGR", "Sharpe Ratio", "Maximum Drawdown"}:
                    a_better = (
                        value_a > value_b
                        if value_a is not None and value_b is not None
                        else None
                    )
                    b_better = (
                        value_b > value_a
                        if value_a is not None and value_b is not None
                        else None
                    )
                else:
                    a_better = (
                        value_a < value_b
                        if value_a is not None and value_b is not None
                        else None
                    )
                    b_better = (
                        value_b < value_a
                        if value_a is not None and value_b is not None
                        else None
                    )
                record[f"A Better: {metric}"] = a_better
                record[f"B Better: {metric}"] = b_better
            pair_rows.append(record)
            records.append(record)

        complete_pair_rows = [
            row
            for row in pair_rows
            if row["Complete Window"] and row["Window Status"] == "complete"
        ]
        aggregate: dict[str, Any] = {
            "Record Type": "aggregate",
            "Reference Variant A": variant_a,
            "Compared Variant B": variant_b,
            "Window ID": None,
            "Start Date": None,
            "End Date": None,
            "Window Status": "complete windows only",
            "Complete Window": True,
            "Number of Complete Windows": sum(
                row["Complete Window"] for row in pair_rows
            ),
        }
        for _metric, delta_name, _scale in metrics:
            values = pd.to_numeric(
                pd.Series([row[delta_name] for row in complete_pair_rows]),
                errors="coerce",
            ).dropna()
            aggregate[f"Mean {delta_name}"] = (
                float(values.mean()) if not values.empty else None
            )
            aggregate[f"Median {delta_name}"] = (
                float(values.median()) if not values.empty else None
            )
        for metric, _delta_name, _scale in metrics:
            wins = [
                row[f"A Better: {metric}"]
                for row in complete_pair_rows
                if row[f"A Better: {metric}"] is not None
            ]
            aggregate[f"A Better Share: {metric}"] = (
                float(pd.Series(wins, dtype=bool).mean()) if wins else None
            )
            aggregate[f"A Better Count: {metric}"] = sum(wins)
            aggregate[f"Comparable Windows: {metric}"] = len(wins)
            b_wins = [
                row[f"B Better: {metric}"]
                for row in complete_pair_rows
                if row[f"B Better: {metric}"] is not None
            ]
            aggregate[f"B Better Share: {metric}"] = (
                float(pd.Series(b_wins, dtype=bool).mean()) if b_wins else None
            )
            aggregate[f"B Better Count: {metric}"] = sum(b_wins)
        records.append(aggregate)
    return pd.DataFrame(records)


def _write_plots(output_dir: Path, details: pd.DataFrame) -> list[str]:
    full = details.loc[
        details["Complete Window"].eq(True) & details["Status"].eq("success")
    ].copy()
    if full.empty:
        return []
    full["Period"] = full["Start Date"].str[:4] + "-" + full["End Date"].str[:4]
    variants = [spec.experiment_id for spec in build_experiment_matrix()]
    plots = (
        ("rolling_3y_net_cagr.html", "Net CAGR", "Net CAGR (%)", 100.0),
        (
            "rolling_3y_maximum_drawdown.html",
            "Maximum Drawdown",
            "Maximum Drawdown (%)",
            100.0,
        ),
        ("rolling_3y_sharpe.html", "Sharpe Ratio", "Sharpe Ratio", 1.0),
    )
    files = []
    for filename, metric, y_title, scale in plots:
        figure = go.Figure()
        for variant in variants:
            rows = full.loc[full["Experiment ID"].eq(variant)]
            figure.add_trace(
                go.Scatter(
                    x=rows["Period"],
                    y=pd.to_numeric(rows[metric], errors="coerce") * scale,
                    mode="lines+markers",
                    name=variant,
                )
            )
        figure.update_layout(
            title=f"Three-year rolling {metric}",
            xaxis_title="Window (end date exclusive)",
            yaxis_title=y_title,
            template="plotly_white",
        )
        figure.write_html(output_dir / filename, include_plotlyjs="directory")
        files.append(filename)

    distribution = go.Figure()
    for variant in variants:
        rows = full.loc[full["Experiment ID"].eq(variant)]
        distribution.add_trace(
            go.Box(
                x=[variant] * len(rows),
                y=pd.to_numeric(rows["Net CAGR"], errors="coerce") * 100,
                name=variant,
                boxpoints="all",
            )
        )
    distribution.update_layout(
        title="Net CAGR distribution across complete windows",
        xaxis_title="Variant",
        yaxis_title="Net CAGR (%)",
        template="plotly_white",
    )
    filename = "rolling_3y_cagr_distribution.html"
    distribution.write_html(output_dir / filename, include_plotlyjs="directory")
    files.append(filename)
    return files


def _window_dependence_note(window_years: int, step_years: int) -> str:
    if step_years < window_years:
        return (
            "Windows overlap and are dependent; standard deviations describe "
            "window dispersion and are not independent-sample estimates."
        )
    if step_years == window_years:
        return (
            "Windows do not overlap. Adjacent chronological windows may still be "
            "serially dependent, so standard deviations are descriptive rather "
            "than independent-sample estimates."
        )
    return (
        "Windows do not overlap and are separated by gaps. Chronological outcomes "
        "may still be serially dependent, so standard deviations are descriptive "
        "rather than independent-sample estimates."
    )


def _report_html(
    run_id: str,
    details: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    plot_files: list[str],
    complete_count: int,
    incomplete_count: int,
    window_years: int,
    step_years: int,
) -> str:
    key_metrics = {
        "Net CAGR",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Average Covariance-Based Portfolio Volatility",
        "Average Exposure",
        "Average Cash Allocation",
        "Transaction Costs",
        "Transaction Costs / Initial Capital",
        "Completed Trade Lots",
        "Average Holding Period (sessions)",
    }
    key_summary = summary.loc[summary["Metric"].isin(key_metrics)]
    detail_columns = [
        "Window ID",
        "Window Status",
        "Experiment ID",
        "Status",
        "Net Total Return",
        "Net CAGR",
        "Gross CAGR",
        "Maximum Drawdown",
        "Sharpe Ratio",
        "Transaction Costs",
        "Average Exposure",
        "Average Cash Allocation",
    ]
    detail_columns = [column for column in detail_columns if column in details]
    pair_summary = comparisons.loc[comparisons["Record Type"].eq("aggregate")]
    signal_c_lines = []
    for variant_a, variant_b in VARIANT_PAIRS[:2]:
        pair = pair_summary.loc[
            pair_summary["Reference Variant A"].eq(variant_a)
            & pair_summary["Compared Variant B"].eq(variant_b)
        ]
        if pair.empty:
            continue
        comparison = pair.iloc[0]
        complete_windows = int(comparison["Number of Complete Windows"])
        cagr_share = comparison.get("B Better Share: Net CAGR")
        sharpe_share = comparison.get("B Better Share: Sharpe Ratio")
        drawdown_share = comparison.get("B Better Share: Maximum Drawdown")
        mean_cagr_delta = comparison.get("Mean CAGR Difference (pp) (B-A)")
        signal_c_lines.append(
            f"{variant_b} vs {variant_a}: mean net CAGR difference "
            f"{mean_cagr_delta:.2f} percentage points; higher net CAGR in "
            f"{cagr_share:.0%}, higher Sharpe in {sharpe_share:.0%}, and "
            f"better maximum drawdown in {drawdown_share:.0%} of "
            f"{complete_windows} complete windows."
            if all(
                pd.notna(value)
                for value in (
                    mean_cagr_delta,
                    cagr_share,
                    sharpe_share,
                    drawdown_share,
                )
            )
            else f"{variant_b} vs {variant_a}: insufficient complete-window metrics."
        )
    worst_rows = []
    for variant, rows in details.loc[
        details["Complete Window"].eq(True) & details["Status"].eq("success")
    ].groupby("Experiment ID", sort=False):
        for metric, label in (
            ("Net CAGR", "Worst CAGR"),
            ("Maximum Drawdown", "Worst drawdown"),
        ):
            numeric = pd.to_numeric(rows[metric], errors="coerce")
            if numeric.notna().any():
                index = numeric.idxmin()
                item = details.loc[index]
                worst_rows.append(
                    {
                        "Variant": variant,
                        "Worst period": label,
                        "Window": item["Window ID"],
                        "Metric": metric,
                        "Value": item[metric],
                    }
                )
    worst = pd.DataFrame(worst_rows)
    plot_sections = "".join(
        f"<h3>{html.escape(filename)}</h3>"
        f"<iframe src='{html.escape(filename)}' title='{html.escape(filename)}' "
        "width='100%' height='520'></iframe>"
        for filename in plot_files
    )
    dependence_note = _window_dependence_note(window_years, step_years)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Three-year rolling robustness</title></head><body>"
        "<h1>Three-year rolling robustness analysis</h1>"
        f"<p>Run ID: {html.escape(run_id)}</p>"
        f"<p>{complete_count} complete and {incomplete_count} incomplete windows. "
        "Only complete windows are included in aggregates and charts; incomplete "
        "windows remain separately identified in detail and pairwise outputs.</p>"
        "<p>Each window starts with fresh initial capital. Window end dates are "
        "exclusive; the last available close before the end is the final valuation. "
        "Indicators may use preceding available history, but returns and portfolio "
        "state are restricted to each window. "
        f"{html.escape(dependence_note)}</p>"
        "<p>Portfolio, trade, cost, and volatility metric definitions follow "
        "<a href='metric_definitions.md'>METRICS.md</a>. Dataset limitations include "
        "survivorship bias because historical index membership is unavailable.</p>"
        "<h2>Complete-window aggregates</h2>"
        f"{key_summary.to_html(index=False, border=0, na_rep='')}"
        "<h2>Signal C stability</h2>"
        f"<ul>{''.join(f'<li>{html.escape(line)}</li>' for line in signal_c_lines)}</ul>"
        "<h2>Worst market periods</h2>"
        f"{worst.to_html(index=False, border=0, na_rep='')}"
        "<h2>Pairwise comparisons (complete windows)</h2>"
        f"{pair_summary.to_html(index=False, border=0, na_rep='')}"
        "<h2>Window detail</h2>"
        f"{details[detail_columns].to_html(index=False, border=0, na_rep='')}"
        "<details><summary>All metrics for all windows</summary>"
        f"{details.to_html(index=False, border=0, na_rep='')}</details>"
        f"<p><a href='rolling_3y_results.csv'>Full detail CSV</a> | "
        "<a href='rolling_3y_results.json'>Full detail JSON</a> | "
        "<a href='rolling_3y_summary.csv'>Summary CSV</a> | "
        "<a href='rolling_3y_comparison.csv'>Pairwise CSV</a> | "
        "<a href='rolling_3y_metadata.json'>Run metadata</a></p>"
        f"<h2>Charts</h2>{plot_sections}</body></html>"
    )


def refresh_rolling_exports(output_dir: Path) -> Path:
    detail_path = output_dir / "rolling_3y_results.csv"
    details = pd.read_csv(detail_path)
    if "Null Metric Reasons" in details.columns:
        details = details.drop(columns="Null Metric Reasons")
    reason_column = pd.Series(
        [_null_metric_reasons(row) for row in details.to_dict(orient="records")],
        name="Null Metric Reasons",
    )
    details = pd.concat([details, reason_column], axis=1)
    details.to_csv(detail_path, index=False)
    (output_dir / "rolling_3y_results.json").write_text(
        json.dumps(_json_safe_records(details), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    metadata_path = output_dir / "rolling_3y_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["detail_fields"] = list(details.columns)
    summary = aggregate_rolling_results(details)
    comparisons = _pairwise_comparisons(details)
    summary.to_csv(output_dir / "rolling_3y_summary.csv", index=False)
    comparisons.to_csv(output_dir / "rolling_3y_comparison.csv", index=False)
    metadata["summary_fields"] = list(summary.columns)
    metadata["comparison_fields"] = list(comparisons.columns)
    metadata["rolling_metrics"] = list(ROLLING_METRICS)
    metadata["overlap_note"] = _window_dependence_note(
        metadata["window_years"], metadata["step_years"]
    )
    metadata["null_metric_reasons"] = (
        "A JSON object in each detail row maps null metric/annual fields to the "
        "documented reason, the absent calendar-year coverage, or the variant error."
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "METRICS.md",
        output_dir / "metric_definitions.md",
    )

    windows = details[["Window ID", "Complete Window"]].drop_duplicates()
    plot_files = [
        filename
        for filename in (
            "rolling_3y_net_cagr.html",
            "rolling_3y_maximum_drawdown.html",
            "rolling_3y_sharpe.html",
            "rolling_3y_cagr_distribution.html",
        )
        if (output_dir / filename).exists()
    ]
    (output_dir / "report.html").write_text(
        _report_html(
            metadata["run_id"],
            details,
            summary,
            comparisons,
            plot_files,
            int(windows["Complete Window"].sum()),
            int(windows["Complete Window"].eq(False).sum()),
            metadata["window_years"],
            metadata["step_years"],
        ),
        encoding="utf-8",
    )
    return output_dir


def run_rolling_comparison(
    portfolio: Portfolio,
    output_root: Path,
    *,
    window_years: int = ROLLING_WINDOW_YEARS,
    step_years: int = ROLLING_STEP_YEARS,
) -> Path:
    close = portfolio.universe.close()
    valid_dates = close.index[close.notna().any(axis=1)]
    if valid_dates.empty:
        raise ValueError("The universe has no usable close-price dates.")
    latest_data_date = pd.Timestamp(valid_dates.max()).normalize()
    windows = build_rolling_windows(
        portfolio.start_date, latest_data_date, window_years, step_years
    )
    specs = build_experiment_matrix()
    cases = tuple((spec.experiment_id, spec.variant, spec.config) for spec in specs)
    initial_capital = portfolio.broker.initial_cash

    result_rows = []
    for window in windows:
        window_portfolio = replace(
            portfolio,
            start_date=window.start_date.date().isoformat(),
            end_date=window.backtest_end_date,
        )
        rows = run_experiment_cases(window_portfolio, cases)
        result_rows.extend(_enrich_result(row, window, initial_capital) for row in rows)

    details = pd.DataFrame(result_rows)
    reason_column = pd.Series(
        [_null_metric_reasons(row) for row in details.to_dict(orient="records")],
        name="Null Metric Reasons",
    )
    details = pd.concat([details, reason_column], axis=1)
    details["Run ID"] = None
    summary = aggregate_rolling_results(details)
    comparisons = _pairwise_comparisons(details)
    available_start = pd.Timestamp(valid_dates.min()).date().isoformat()
    available_end_exclusive = (
        (latest_data_date + pd.Timedelta(days=1)).date().isoformat()
    )
    fingerprints = details.get("Universe Fingerprint", pd.Series(dtype=object))
    fingerprints = fingerprints.dropna().drop_duplicates()
    parameters: dict[str, Any] = {
        "run_type": "rolling_robustness",
        "start_date": portfolio.start_date,
        "available_data_start_date": available_start,
        "latest_data_date": latest_data_date.date().isoformat(),
        "available_data_end_date_exclusive": available_end_exclusive,
        "window_years": window_years,
        "step_years": step_years,
        "end_date_semantics": "exclusive",
        "incomplete_windows_are_excluded_from_aggregates": True,
        "initial_capital_eur_per_window": initial_capital,
        "broker": asdict(portfolio.broker),
        "universe_size": len(close.columns),
        "universe_fingerprint": (
            fingerprints.iloc[0] if not fingerprints.empty else None
        ),
        "variants": [
            {
                "experiment_id": spec.experiment_id,
                "signals": spec.variant.value,
                "config": asdict(spec.config),
            }
            for spec in specs
        ],
        "metric_aliases": METRIC_ALIASES,
        "metric_definitions": "metric_definitions.md",
        "rolling_metrics": list(ROLLING_METRICS),
        "detail_fields": list(details.columns),
        "summary_fields": list(summary.columns),
        "comparison_fields": list(comparisons.columns),
        "null_metric_reasons": (
            "A JSON object in each detail row maps null metric/annual fields to "
            "the documented reason, absent calendar-year coverage, or variant error."
        ),
        "window_completeness_rule": (
            "A window is complete when its nominal exclusive end is on or before "
            "the day after the latest date with any close observation in the local "
            "universe. Incomplete windows are run only through the latest observed "
            "date and excluded from aggregates."
        ),
        "windows": [
            {
                "window_id": window.window_id,
                "start_date": window.start_date.date().isoformat(),
                "end_date_exclusive": window.end_date_exclusive.date().isoformat(),
                "observed_end_date_exclusive": (
                    window.observed_end_exclusive.date().isoformat()
                ),
                "backtest_end_date_inclusive": window.backtest_end_date,
                "status": "complete" if window.complete else "incomplete",
            }
            for window in windows
        ],
        "portfolio_state": (
            "Each window is run with a fresh VectorBT portfolio and the same "
            "configured initial cash; holdings are never carried between windows."
        ),
        "lookback_policy": (
            "The existing momentum preparation receives the local close history "
            "before the window for causal indicator warm-up; simulation prices and "
            "returns are sliced to [start_date, observed_end_date_exclusive)."
        ),
        "overlap_note": _window_dependence_note(window_years, step_years),
        "data_limitations": (
            "Instrument files lack historical index membership and may introduce "
            "survivorship bias. Completeness is dataset-level; instruments can "
            "have different exchange calendars or missing observations."
        ),
        "git_revision": _git_revision(Path(__file__).resolve().parents[2]),
        "created_at": datetime.now(UTC).isoformat(),
    }
    digest = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"rolling-3y-{timestamp}-{digest}"
    parameters["run_id"] = run_id
    details["Run ID"] = run_id
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    details.to_csv(output_dir / "rolling_3y_results.csv", index=False)
    (output_dir / "rolling_3y_results.json").write_text(
        json.dumps(_json_safe_records(details), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary.to_csv(output_dir / "rolling_3y_summary.csv", index=False)
    comparisons.to_csv(output_dir / "rolling_3y_comparison.csv", index=False)
    (output_dir / "rolling_3y_metadata.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True), encoding="utf-8"
    )
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "METRICS.md",
        output_dir / "metric_definitions.md",
    )
    plot_files = _write_plots(output_dir, details)
    complete_count = sum(window.complete for window in windows)
    incomplete_count = len(windows) - complete_count
    (output_dir / "report.html").write_text(
        _report_html(
            run_id,
            details,
            summary,
            comparisons,
            plot_files,
            complete_count,
            incomplete_count,
            window_years,
            step_years,
        ),
        encoding="utf-8",
    )
    return output_dir


def run_standard_rolling_comparison(
    start_date: str = DEFAULT_START_DATE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    output_root: Path | None = None,
    step_years: int = ROLLING_STEP_YEARS,
) -> Path:
    from pytrading.data_helper import Universe

    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    project_root = Path(__file__).resolve().parents[2]
    universe = Universe()
    close = universe.close()
    valid_dates = close.index[close.notna().any(axis=1)]
    if valid_dates.empty:
        raise ValueError("The universe has no usable close-price dates.")
    portfolio = Portfolio(
        universe=universe,
        start_date=start_date,
        end_date=pd.Timestamp(valid_dates.max()).date().isoformat(),
        broker=default_broker_config(initial_capital),
        benchmarks={},
    )
    return run_rolling_comparison(
        portfolio,
        output_root or project_root / "results" / "experiments",
        step_years=step_years,
    )
