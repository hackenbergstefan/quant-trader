from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from pytrading import BrokerConfig, Portfolio
from pytrading.strategies.momentum import (
    MomentumConfig,
    StrategyVariant,
    run_experiment_cases,
)

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2017-01-01"
DEFAULT_INITIAL_CAPITAL = 3000.0
DEFAULT_FEES = 0.001
DEFAULT_FIXED_FEES = 1.0
DEFAULT_SLIPPAGE = 0.0005


def default_experiment_config() -> MomentumConfig:
    return MomentumConfig(
        topn=5,
        min_avg_daily_turnover=50_000.0,
        liquidity_lookback=20,
        momentum_20_weight=0.2,
        momentum_60_weight=0.3,
        momentum_120_weight=0.5,
        rebalance_frequency="monthly",
        max_position_weight=0.2,
    )


def default_broker_config(
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> BrokerConfig:
    return BrokerConfig(
        initial_cash=initial_capital,
        fees=DEFAULT_FEES,
        fixed_fees=DEFAULT_FIXED_FEES,
        slippage=DEFAULT_SLIPPAGE,
    )


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    variant: StrategyVariant
    config: MomentumConfig


def build_experiment_matrix(
    base_config: MomentumConfig | None = None,
) -> tuple[ExperimentSpec, ...]:
    base_config = base_config or default_experiment_config()
    return (
        ExperimentSpec(
            "AB-M5",
            StrategyVariant.AB,
            replace(base_config, topn=5, rebalance_frequency="monthly"),
        ),
        ExperimentSpec(
            "ABC-M5",
            StrategyVariant.ABC,
            replace(base_config, topn=5, rebalance_frequency="monthly"),
        ),
        ExperimentSpec(
            "AB-M10",
            StrategyVariant.AB,
            replace(base_config, topn=10, rebalance_frequency="monthly"),
        ),
        ExperimentSpec(
            "ABC-M10",
            StrategyVariant.ABC,
            replace(base_config, topn=10, rebalance_frequency="monthly"),
        ),
    )


def load_experiment_config(
    config_path: Path,
) -> tuple[dict[str, Any], tuple[ExperimentSpec, ...]]:
    archived = json.loads(config_path.read_text(encoding="utf-8"))
    specs = tuple(
        ExperimentSpec(
            experiment_id=item["experiment_id"],
            variant=StrategyVariant(item["signals"]),
            config=MomentumConfig(
                **{
                    key: value
                    for key, value in item["config"].items()
                    if key != "position_sizing"
                }
            ),
        )
        for item in archived["variants"]
    )
    return archived, specs


def _yearly_returns(values: pd.Series, initial_capital: float) -> pd.Series:
    year_end_values = values.groupby(values.index.year).last()
    returns = {}
    previous_value = initial_capital
    for year, year_end_value in year_end_values.items():
        returns[int(year)] = year_end_value / previous_value - 1
        previous_value = year_end_value
    return pd.Series(returns, dtype=float)


def _run_id(parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"compare-{timestamp}-{digest}"


def _git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.to_dict(orient="records"):
        safe_row = {}
        for key, value in row.items():
            if pd.isna(value):
                value = None
            elif isinstance(value, float) and not isfinite(value):
                value = "Infinity" if value > 0 else "-Infinity"
            elif hasattr(value, "item"):
                value = value.item()
            safe_row[key] = value
        records.append(safe_row)
    return records


def _write_plots(
    output_dir: Path,
    results: list[dict[str, Any]],
    initial_capital: float,
) -> list[str]:
    successful = [row for row in results if row.get("Status") == "success"]
    if not successful:
        return []

    net_series = {
        row["Experiment ID"]: row["_Net Equity"]
        for row in successful
        if isinstance(row.get("_Net Equity"), pd.Series)
    }
    gross_series = {
        row["Experiment ID"]: row["_Gross Equity"]
        for row in successful
        if isinstance(row.get("_Gross Equity"), pd.Series)
    }
    curve_frame = pd.concat(
        {
            **{f"{name} Net": values for name, values in net_series.items()},
            **{f"{name} Gross": values for name, values in gross_series.items()},
        },
        axis=1,
    )
    curve_frame.index.name = "Date"
    curve_frame.to_csv(output_dir / "equity_curves.csv")

    plots: list[tuple[str, str, dict[str, pd.Series], str]] = []
    plots.append(("equity_net.html", "Net Equity Curves", net_series, "EUR"))
    plots.append(("equity_gross.html", "Gross Equity Curves", gross_series, "EUR"))
    drawdowns = {
        name: values.div(values.cummax()).sub(1) for name, values in net_series.items()
    }
    plots.append(("drawdowns.html", "Net Drawdowns", drawdowns, "Drawdown"))
    yearly_net_returns = {
        row["Experiment ID"]: _yearly_returns(row["_Net Equity"], initial_capital)
        for row in successful
    }
    plots.append(
        ("annual_net_returns.html", "Annual Net Returns", yearly_net_returns, "Return")
    )
    yearly_turnover = {
        row["Experiment ID"]: pd.Series(
            {
                int(str(key).rsplit(" ", 1)[1]): value
                for key, value in row.items()
                if str(key).startswith("Annualized Two-sided Turnover ")
            },
            dtype=float,
        )
        for row in successful
    }
    plots.append(
        (
            "annual_turnover.html",
            "Annualized Two-sided Turnover",
            yearly_turnover,
            "Turnover",
        )
    )
    yearly_costs = {
        row["Experiment ID"]: pd.Series(
            {
                int(str(key).rsplit(" ", 1)[1]): value
                for key, value in row.items()
                if str(key).startswith("Annual Transaction Costs ")
            },
            dtype=float,
        )
        for row in successful
    }
    plots.append(
        (
            "annual_transaction_costs.html",
            "Annual Transaction Costs",
            yearly_costs,
            "EUR",
        )
    )

    plot_files = []
    for filename, title, series_by_name, y_title in plots:
        figure = go.Figure()
        for name, series in series_by_name.items():
            if not series.empty:
                figure.add_trace(
                    go.Scatter(x=series.index.astype(str), y=series, name=name)
                )
        figure.update_layout(
            title=title,
            xaxis_title="Date / Year",
            yaxis_title=y_title,
            hovermode="x unified",
            template="plotly_white",
        )
        figure.write_html(output_dir / filename, include_plotlyjs="directory")
        plot_files.append(filename)

    position_figure = go.Figure()
    for row in successful:
        for key, label, dash in (
            ("_Position Weight Average", "average", "solid"),
            ("_Position Weight Maximum", "maximum", "dot"),
        ):
            series = row.get(key)
            if isinstance(series, pd.Series):
                position_figure.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series,
                        name=f"{row['Experiment ID']} {label}",
                        line={"dash": dash},
                    )
                )
    position_cap = float(successful[0].get("max_position_weight", 0.2))
    position_figure.add_hline(
        y=position_cap,
        line_dash="dash",
        annotation_text=f"Target cap {position_cap:.0%}",
    )
    position_figure.update_layout(
        title="Daily Average and Largest Position Weights",
        xaxis_title="Date",
        yaxis_title="Share of net portfolio value",
        hovermode="x unified",
        template="plotly_white",
    )
    position_figure.write_html(
        output_dir / "position_weights.html", include_plotlyjs="directory"
    )
    plot_files.append("position_weights.html")
    return plot_files


def _interpret_results(results: list[dict[str, Any]]) -> list[str]:
    successful = [row for row in results if row.get("Status") == "success"]
    if not successful:
        return ["No experiment variant completed successfully."]

    lines = []
    best_net = max(successful, key=lambda row: row["Net Total Return"])
    least_drawdown = max(successful, key=lambda row: row["Maximum Drawdown"])
    lowest_turnover = min(
        successful, key=lambda row: row["Annualized Two-sided Turnover"]
    )
    best_sharpe = max(successful, key=lambda row: row["Sharpe"])
    lines.extend(
        [
            f"Highest net total return: {best_net['Experiment ID']} ({best_net['Net Total Return']:.2%}).",
            f"Smallest maximum drawdown: {least_drawdown['Experiment ID']} ({least_drawdown['Maximum Drawdown']:.2%}).",
            f"Lowest annualized two-sided turnover: {lowest_turnover['Experiment ID']} ({lowest_turnover['Annualized Two-sided Turnover']:.2f}x).",
            f"Highest Sharpe ratio: {best_sharpe['Experiment ID']} ({best_sharpe['Sharpe']:.3f}); winner is not selected by CAGR alone.",
        ]
    )
    lines.append(
        "The 20% cap applies to rebalance target weights; market moves can push "
        "actual daily position weights above it until the next monthly rebalance."
    )
    for row in successful:
        lines.append(
            f"{row['Experiment ID']}: gross/net return gap "
            f"{(row['Gross Total Return'] - row['Net Total Return']) * 100:.2f} "
            "percentage points."
        )

    by_id = {row["Experiment ID"]: row for row in successful}

    def delta(
        first: dict[str, Any],
        second: dict[str, Any],
        key: str,
        *,
        scale: float = 1.0,
        suffix: str = "",
    ) -> str:
        first_value, second_value = first.get(key), second.get(key)
        if not pd.notna(first_value) or not pd.notna(second_value):
            return "n/a"
        return f"{(float(second_value) - float(first_value)) * scale:+.2f}{suffix}"

    for topn in (5, 10):
        equal_weighted = by_id.get(f"AB-M{topn}")
        inverse_volatility = by_id.get(f"ABC-M{topn}")
        if equal_weighted and inverse_volatility:
            lines.append(
                f"Signal C at Top {topn} (ABC-M{topn} vs AB-M{topn}): "
                f"net return change {delta(equal_weighted, inverse_volatility, 'Net Total Return', scale=100, suffix=' pp')}; "
                f"Sharpe change {delta(equal_weighted, inverse_volatility, 'Sharpe')}; "
                f"realized portfolio volatility change {delta(equal_weighted, inverse_volatility, 'Average Realized Volatility', scale=100, suffix=' pp')}; "
                f"covariance-based portfolio volatility change {delta(equal_weighted, inverse_volatility, 'Average Covariance-Based Portfolio Volatility', scale=100, suffix=' pp')}; "
                f"weight-weighted constituent volatility change {delta(equal_weighted, inverse_volatility, 'Weight-Weighted Average Constituent Volatility', scale=100, suffix=' pp')}; "
                f"annualized turnover change {delta(equal_weighted, inverse_volatility, 'Annualized Two-sided Turnover', suffix='x')}; "
                f"transaction cost change {delta(equal_weighted, inverse_volatility, 'Transaction Costs', suffix=' EUR')}."
            )

    for signal_prefix in ("AB", "ABC"):
        top5, top10 = (
            by_id.get(f"{signal_prefix}-M5"),
            by_id.get(f"{signal_prefix}-M10"),
        )
        if top5 and top10:
            lines.append(
                f"Top 10 vs Top 5 for {signal_prefix}: "
                f"net return change {delta(top5, top10, 'Net Total Return', scale=100, suffix=' pp')}; "
                f"Sharpe change {delta(top5, top10, 'Sharpe')}; "
                f"maximum drawdown change {delta(top5, top10, 'Maximum Drawdown', scale=100, suffix=' pp')}; "
                f"annualized turnover change {delta(top5, top10, 'Annualized Two-sided Turnover', suffix='x')}; "
                f"transaction cost change {delta(top5, top10, 'Transaction Costs', suffix=' EUR')}."
            )
    failures = [row for row in results if row.get("Status") != "success"]
    if failures:
        lines.append(
            f"Incomplete variants: {', '.join(row['Experiment ID'] for row in failures)}."
        )
    return lines


def run_comparison(
    portfolio: Portfolio,
    output_root: Path,
    *,
    base_config: MomentumConfig | None = None,
    run_id: str | None = None,
) -> Path:
    specs = build_experiment_matrix(base_config)
    parameters = {
        "start_date": portfolio.start_date,
        "end_date": portfolio.end_date,
        "broker": asdict(portfolio.broker),
        "currency": "EUR",
        "signal_d": "excluded",
        "signal_c_semantics": (
            "Signal C alone enables inverse-volatility weighting using equal "
            "parts realized volatility over 20 and 60 sessions; without C, "
            "selected positions are equal weighted."
        ),
        "position_cap_semantics": (
            "The 20% cap applies to target weights at monthly rebalances; actual "
            "weights can drift above the cap between rebalances."
        ),
        "variants": [
            {
                "experiment_id": spec.experiment_id,
                "signals": spec.variant.value,
                "weighting": (
                    "inverse_volatility" if "C" in spec.variant.value else "equal"
                ),
                "config": asdict(spec.config),
            }
            for spec in specs
        ],
    }
    run_id = run_id or _run_id(parameters)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = run_experiment_cases(
        portfolio,
        tuple((spec.experiment_id, spec.variant, spec.config) for spec in specs),
    )
    parameters.update(
        {
            "run_id": run_id,
            "git_revision": _git_revision(Path(__file__).resolve().parents[2]),
            "universe_size": len(portfolio.universe.close().columns),
            "universe_limitation": (
                "Instrument files have no historical index-membership dates; "
                "results may contain survivorship bias and are not survivorship-bias-free."
            ),
            "price_treatment": (
                "Close is adjusted; Open/High/Low are scaled by Close/Raw Close; "
                "all price and turnover fields are converted to EUR using prior FX rates."
            ),
            "metric_definitions": "metric_definitions.md",
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True), encoding="utf-8"
    )

    flat_rows = [
        {key: value for key, value in row.items() if not str(key).startswith("_")}
        for row in rows
    ]
    result_frame = pd.DataFrame(flat_rows)
    result_frame.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "results.json").write_text(
        json.dumps(_json_safe_records(result_frame), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "METRICS.md",
        output_dir / "metric_definitions.md",
    )
    plot_files = _write_plots(output_dir, rows, portfolio.broker.initial_cash)
    interpretation = _interpret_results(rows)
    successful_count = sum(row.get("Status") == "success" for row in rows)
    table = result_frame.to_html(index=False, border=0, na_rep="")
    plot_links = "\n".join(
        f'<li><a href="{name}">{name}</a></li>' for name in plot_files
    )
    report_html = (
        "<!doctype html><meta charset='utf-8'><title>Strategy comparison</title>"
        "<h1>Strategy comparison</h1>"
        f"<p>{successful_count}/{len(rows)} variants completed successfully.</p>"
        f"<p>Run ID: {run_id}</p><ul>{plot_links}</ul>"
        "<p><a href='results.json'>JSON results</a> | "
        "<a href='metric_definitions.md'>Metric definitions</a></p>"
        f"<h2>Results</h2>{table}"
        f"<h2>Interpretation</h2><ul>{''.join(f'<li>{line}</li>' for line in interpretation)}</ul>"
        f"<h2>Limitations</h2><p>{parameters['universe_limitation']}</p>"
    )
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")
    report_markdown = [
        "# Momentum strategy comparison",
        "",
        f"Run: `{run_id}`",
        f"Completed: {successful_count}/{len(rows)} variants",
        "",
        "## Interpretation",
        *[f"- {line}" for line in interpretation],
        "",
        "## Limitations",
        parameters["universe_limitation"],
        "",
        "## Files",
        "- `results.csv`",
        "- `results.json` (same result fields as `results.csv`)",
        "- `equity_curves.csv` (when at least one variant succeeds)",
        "- `config.json`",
        "- `metric_definitions.md`",
        *[f"- `{name}`" for name in plot_files],
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report_markdown) + "\n", encoding="utf-8"
    )
    logger.info("Experimentbericht gespeichert: %s", output_dir)
    return output_dir


def run_standard_comparison(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    output_root: Path | None = None,
) -> Path:
    from pytrading.data_helper import Universe

    if end_date <= start_date:
        raise ValueError("End date must be after start date.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    project_root = Path(__file__).resolve().parents[2]
    portfolio = Portfolio(
        universe=Universe(),
        start_date=start_date,
        end_date=end_date,
        broker=default_broker_config(initial_capital),
        benchmarks={},
    )
    return run_comparison(
        portfolio, output_root or project_root / "results" / "experiments"
    )
