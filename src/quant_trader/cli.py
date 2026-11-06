from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer

from quant_trader.backtesting.benchmark import buy_and_hold_benchmark
from quant_trader.config import AppConfig, RiskConfig, load_config
from quant_trader.data.calendars import ExchangeTradingCalendar
from quant_trader.data.corporate_actions import actions_from_frame
from quant_trader.data.instruments import YamlInstrumentRepository
from quant_trader.data.providers import YahooFinanceProvider
from quant_trader.data.storage import MarketDataStore
from quant_trader.data.synthetic import synthetic_market_data
from quant_trader.data.update import update_history
from quant_trader.domain.models import Instrument
from quant_trader.experiments.runner import (
    run_comparison,
    run_cost_sensitivity,
    run_parameter_sensitivity,
    run_variant,
)
from quant_trader.experiments.walk_forward import run_holdout, run_walk_forward
from quant_trader.reporting.charts import write_charts
from quant_trader.reporting.notebooks import write_report_notebook
from quant_trader.reporting.reports import configuration_hash, data_hash, write_report

app = typer.Typer(help="European equity research and backtesting", no_args_is_help=True)
data_app = typer.Typer(help="Market data operations", no_args_is_help=True)
backtest_app = typer.Typer(help="Backtest operations", no_args_is_help=True)
report_app = typer.Typer(help="Report operations", no_args_is_help=True)
experiment_app = typer.Typer(help="Run strategy comparisons and reports", no_args_is_help=True)
experiments_app = typer.Typer(help="Validation and sensitivity experiments", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(report_app, name="report")
app.add_typer(experiment_app, name="experiment")
app.add_typer(experiments_app, name="experiments")

DEFAULT_SYNTHETIC_START = date(2010, 1, 1)
DEFAULT_SYNTHETIC_END = date(2024, 12, 31)
BENCHMARK_ID = "EUROSTOXX50"


def _read_config(config_dir: Path) -> AppConfig:
    return load_config(config_dir)


def _parse_date(value: str | None, option: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must use YYYY-MM-DD") from exc


def _benchmark_instrument(config: AppConfig) -> Instrument:
    return Instrument(
        BENCHMARK_ID,
        config.data.benchmark_symbol,
        config.data.benchmark_exchange,
        config.data.benchmark_currency,
        config.data.benchmark_calendar,
    )


def _load_data(
    data_source: Literal["synthetic", "stored"],
    config: AppConfig,
    *,
    config_dir: Path,
    end: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list, str]:
    if data_source == "synthetic":
        bars, reference = synthetic_market_data(
            DEFAULT_SYNTHETIC_START, end or DEFAULT_SYNTHETIC_END
        )
        return bars, reference, [], "deterministic synthetic data (not historical observations)"

    database = Path(config.data.database)
    if not database.is_absolute():
        database = config_dir.parent / database
    with MarketDataStore(database) as store:
        prices = store.load_prices()
        action_frame = store.load_actions()
    if end is not None:
        prices = prices.loc[pd.to_datetime(prices["timestamp"]).dt.date <= end]
        action_frame = action_frame.loc[pd.to_datetime(action_frame["ex_date"]).dt.date <= end]
    if prices.empty:
        raise typer.BadParameter(
            f"No stored prices in {database}; run `quant-trader data update` first"
        )
    reference = prices.loc[prices["instrument_id"] == BENCHMARK_ID].copy()
    bars = prices.loc[prices["instrument_id"] != BENCHMARK_ID].copy()
    if reference.empty or bars.empty:
        raise typer.BadParameter(
            "Stored data must contain equity bars and the EURO STOXX 50 benchmark"
        )
    return bars, reference, actions_from_frame(action_frame), "DuckDB normalized Yahoo Finance data"


def _risk_override(
    config: AppConfig, capital: float | None, allocation_method: str | None
) -> RiskConfig:
    values = config.risk.model_dump()
    if capital is not None:
        values["starting_cash_eur"] = capital
    if allocation_method is not None:
        values["allocation_method"] = allocation_method
    return RiskConfig.model_validate(values)


def _report_data(bars: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat([bars, reference], ignore_index=True)
        .sort_values(["timestamp", "instrument_id"], kind="stable")
        .reset_index(drop=True)
    )


def _output_dir(
    strategy: str,
    config: AppConfig,
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir
    name = strategy.lower().replace(" ", "_")
    return (
        Path("data/backtests")
        / f"{name}_{configuration_hash(config)[:8]}_{data_hash(_report_data(bars, reference))[:8]}"
    )


def _write_experiment(
    table: pd.DataFrame,
    root: Path,
    name: str,
    config: AppConfig,
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    data_source: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / f"{name}.csv", index=False)
    metadata = {
        "experiment": name,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "configuration_hash": configuration_hash(config),
        "data_hash": data_hash(_report_data(bars, reference)),
        "data_source": data_source,
        "coverage_start": pd.to_datetime(bars["timestamp"]).min().date().isoformat(),
        "coverage_end": pd.to_datetime(bars["timestamp"]).max().date().isoformat(),
        "configuration": config.model_dump(mode="json"),
        "results": table.replace({float("nan"): None}).to_dict(orient="records"),
    }
    (root / f"{name}.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"
    )
    columns = list(table.columns)
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join("N/A" if pd.isna(value) else str(value) for value in row) + " |"
        )
    (root / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_comparison_workflow(
    *,
    data_source: Literal["synthetic", "stored"],
    end_date: date | None,
    output_dir: Path,
    config_dir: Path,
    generate_reports: bool,
) -> pd.DataFrame:
    config = _read_config(config_dir)
    bars, reference, actions, source = _load_data(
        data_source, config, config_dir=config_dir, end=end_date
    )
    results, table = run_comparison(bars, reference, config, actions)
    _write_experiment(table, output_dir, "comparison", config, bars, reference, source)
    report_bars = _report_data(bars, reference)
    for name, result in results.items():
        result.equity_curve.to_csv(output_dir / f"{name}_equity_curve.csv", index=False)
    if generate_reports:
        reports_dir = output_dir / "reports"
        for name, result in results.items():
            write_report(
                result,
                config,
                report_bars,
                reports_dir / name,
                data_source=source,
                additional_metadata={
                    "experiment": "strategy comparison",
                    "variant": name,
                    "universe_method": "configured current research seed; not a point-in-time historical universe",
                    "historical_universe_bias": "Results on this current seed are survivorship-biased and not market-wide evidence.",
                },
            )
        typer.echo(f"Variant reports: {reports_dir}")
    typer.echo(f"Comparison: {output_dir / 'comparison.md'}")
    typer.echo(table.to_string(index=False))
    return table


@data_app.command("update")
def data_update(
    start: str = typer.Option("2010-01-01", help="First daily session to fetch (YYYY-MM-DD)"),
    end: str = typer.Option(
        date.today().isoformat(), help="Last daily session to fetch (YYYY-MM-DD)"
    ),
    instrument_ids: str | None = typer.Option(
        None, help="Comma-separated configured instrument IDs; default is the full universe"
    ),
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    start_date = _parse_date(start, "--start")
    end_date = _parse_date(end, "--end")
    config = _read_config(config_dir)
    repository = YamlInstrumentRepository(config_dir / "universe.yaml")
    selected_ids = (
        [value.strip() for value in instrument_ids.split(",") if value.strip()]
        if instrument_ids
        else None
    )
    database = Path(config.data.database)
    if not database.is_absolute():
        database = config_dir.parent / database
    with MarketDataStore(database) as store:
        rows, failures = update_history(
            repository,
            YahooFinanceProvider(),
            store,
            start_date,
            end_date,
            _benchmark_instrument(config),
            selected_ids,
        )
    typer.echo(f"Stored {rows:,} normalized bars in {database}.")
    if failures:
        typer.echo(f"Failed instruments ({len(failures)}): {', '.join(failures)}", err=True)
        raise typer.Exit(1)


@data_app.command("validate")
def data_validate(
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    config = _read_config(config_dir)
    database = Path(config.data.database)
    if not database.is_absolute():
        database = config_dir.parent / database
    with MarketDataStore(database) as store:
        prices = store.load_prices()
    if prices.empty:
        raise typer.BadParameter(f"No stored prices in {database}")
    calendar_service = ExchangeTradingCalendar()
    results = []
    for instrument_id, group in prices.groupby("instrument_id", sort=True):
        start = pd.to_datetime(group["timestamp"]).min().date()
        end = pd.to_datetime(group["timestamp"]).max().date()
        calendar = str(group["calendar"].iloc[0])
        sessions = calendar_service.sessions(calendar, start, end)
        observed = pd.DatetimeIndex(pd.to_datetime(group["timestamp"])).normalize()
        missing = sessions.difference(observed)
        results.append(
            {
                "instrument_id": instrument_id,
                "calendar": calendar,
                "bars": len(group),
                "missing_sessions": len(missing),
                "missing_dates": [stamp.date().isoformat() for stamp in missing],
            }
        )
    typer.echo(
        json.dumps(
            {"database": str(database), "instruments": len(results), "results": results}, indent=2
        )
    )
    if any(item["missing_sessions"] for item in results):
        raise typer.Exit(1)


@app.command("universe")
def universe_show(
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    repository = YamlInstrumentRepository(config_dir / "universe.yaml")
    instruments = repository.list_instruments()
    typer.echo(f"Configured research seed: {len(instruments)} active equities")
    typer.echo("instrument_id\texchange\tcurrency\tcalendar")
    for item in instruments:
        typer.echo(f"{item.instrument_id}\t{item.exchange}\t{item.currency}\t{item.calendar}")


@backtest_app.command("run")
def backtest_run(
    strategy: str = typer.Option(
        "adaptive_momentum", help="Strategy: A, B, C, D, or adaptive_momentum"
    ),
    data_source: Literal["synthetic", "stored"] = typer.Option(
        "synthetic", help="Offline synthetic or stored provider data"
    ),
    start: str | None = typer.Option(None, help="First portfolio valuation date (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="Last portfolio valuation date (YYYY-MM-DD)"),
    capital: float | None = typer.Option(None, help="Override starting EUR cash"),
    allocation_method: str | None = typer.Option(
        None, help="equal_weight, volatility_adjusted, or risk_budget"
    ),
    output_dir: Path | None = typer.Option(None, help="Report output directory"),
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    start_date = _parse_date(start, "--start")
    end_date = _parse_date(end, "--end")
    config = _read_config(config_dir)
    risk = _risk_override(config, capital, allocation_method)
    config = config.model_copy(update={"risk": risk})
    bars, reference, actions, source = _load_data(
        data_source, config, config_dir=config_dir, end=end_date
    )
    if strategy.upper() in {"A", "BENCHMARK", "EURO_STOXX_50"}:
        benchmark_data = reference
        if start_date is not None:
            benchmark_data = benchmark_data.loc[
                pd.to_datetime(benchmark_data["timestamp"]).dt.date >= start_date
            ]
        result = buy_and_hold_benchmark(
            benchmark_data, risk.starting_cash_eur, config.execution.total_cost_rate
        )
        strategy_name = "A_euro_stoxx_50_buy_and_hold"
    else:
        result = run_variant(
            strategy,
            bars,
            reference,
            config.strategy,
            risk,
            config.execution,
            actions,
            start_date=start_date,
            end_date=end_date,
        )
        strategy_name = result.metadata["strategy"]
    report_bars = _report_data(bars, reference)
    root = _output_dir(strategy_name, config, bars, reference, output_dir)
    path = write_report(
        result,
        config,
        report_bars,
        root,
        data_source=source,
        additional_metadata={
            "universe_method": "configured current research seed; not a point-in-time historical universe",
            "historical_universe_bias": "Results on this current seed are survivorship-biased and not market-wide evidence.",
        },
    )
    typer.echo(f"Report: {path}")
    typer.echo(json.dumps(result.metrics, indent=2, default=str))


@backtest_app.command("compare")
def backtest_compare(
    data_source: Literal["synthetic", "stored"] = typer.Option(
        "synthetic", help="Offline synthetic or stored provider data"
    ),
    end: str | None = typer.Option(None, help="Last data date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(
        Path("data/backtests/comparison"), help="Comparison output directory"
    ),
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    end_date = _parse_date(end, "--end")
    _run_comparison_workflow(
        data_source=data_source,
        end_date=end_date,
        output_dir=output_dir,
        config_dir=config_dir,
        generate_reports=False,
    )


@experiment_app.command("run")
def experiment_run(
    data_source: Annotated[
        Literal["synthetic", "stored"],
        typer.Option(help="Offline synthetic or stored provider data"),
    ] = "synthetic",
    end: Annotated[str | None, typer.Option(help="Last data date (YYYY-MM-DD)")] = None,
    report: Annotated[
        bool,
        typer.Option(
            "--report/--no-report", help="Generate a full report for every comparison variant"
        ),
    ] = False,
    output_dir: Annotated[Path, typer.Option(help="Comparison and report output directory")] = Path(
        "data/backtests/experiment"
    ),
    config_dir: Annotated[Path, typer.Option(help="Configuration directory")] = Path("config"),
) -> None:
    _run_comparison_workflow(
        data_source=data_source,
        end_date=_parse_date(end, "--end"),
        output_dir=output_dir,
        config_dir=config_dir,
        generate_reports=report,
    )


@report_app.command("generate")
def report_generate(
    run_dir: Path = typer.Option(..., help="Directory containing report.json and equity_curve.csv"),
) -> None:
    report_path = run_dir / "report.json"
    equity_path = run_dir / "equity_curve.csv"
    if not report_path.exists() or not equity_path.exists():
        raise typer.BadParameter("Run directory must contain report.json and equity_curve.csv")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    write_charts(pd.read_csv(equity_path), run_dir, report["metadata"].get("strategy", "Backtest"))
    notebook_path = write_report_notebook(report, run_dir)
    typer.echo(f"Charts and notebook regenerated in {run_dir}; notebook: {notebook_path}")


@experiments_app.command("walk-forward")
def experiments_walk_forward(
    data_source: Literal["synthetic", "stored"] = typer.Option("synthetic", help="Data source"),
    training_years: int = typer.Option(5, min=1),
    test_years: int = typer.Option(2, min=1),
    step_years: int = typer.Option(1, min=1),
    end: str | None = typer.Option(None, help="Last data date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("data/backtests/walk_forward"), help="Output directory"),
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    end_date = _parse_date(end, "--end")
    config = _read_config(config_dir)
    bars, reference, actions, source = _load_data(
        data_source, config, config_dir=config_dir, end=end_date
    )
    table = run_walk_forward(
        bars,
        reference,
        config,
        actions=actions,
        training_years=training_years,
        test_years=test_years,
        step_years=step_years,
    )
    _write_experiment(table, output_dir, "walk_forward", config, bars, reference, source)
    typer.echo(f"Walk-forward windows: {len(table)}; report: {output_dir / 'walk_forward.md'}")


@experiments_app.command("holdout")
def experiments_holdout(
    data_source: Literal["synthetic", "stored"] = typer.Option("synthetic", help="Data source"),
    end: str | None = typer.Option(None, help="Last available data date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("data/backtests/holdout"), help="Output directory"),
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    end_date = _parse_date(end, "--end")
    config = _read_config(config_dir)
    bars, reference, actions, source = _load_data(
        data_source, config, config_dir=config_dir, end=end_date
    )
    table = run_holdout(bars, reference, config, actions=actions)
    _write_experiment(table, output_dir, "holdout", config, bars, reference, source)
    typer.echo(f"Holdout results: {output_dir / 'holdout.md'}")


@experiments_app.command("sensitivity")
def experiments_sensitivity(
    data_source: Literal["synthetic", "stored"] = typer.Option("synthetic", help="Data source"),
    end: str | None = typer.Option(None, help="Last available data date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("data/backtests/sensitivity"), help="Output directory"),
    config_dir: Path = typer.Option(Path("config"), help="Configuration directory"),
) -> None:
    end_date = _parse_date(end, "--end")
    config = _read_config(config_dir)
    bars, reference, actions, source = _load_data(
        data_source, config, config_dir=config_dir, end=end_date
    )
    table = run_parameter_sensitivity(bars, reference, config, actions=actions)
    _write_experiment(table, output_dir, "parameter_sensitivity", config, bars, reference, source)
    costs = run_cost_sensitivity(bars, reference, config, actions=actions)
    _write_experiment(costs, output_dir, "cost_sensitivity", config, bars, reference, source)
    typer.echo(f"Parameter and cost sensitivity reports: {output_dir}")


if __name__ == "__main__":
    app()
