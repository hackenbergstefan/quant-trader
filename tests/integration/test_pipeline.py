import json
from datetime import date

import pandas as pd

from quant_trader.config import AppConfig, ExecutionConfig, RiskConfig, StrategyConfig
from quant_trader.data.synthetic import synthetic_market_data
from quant_trader.domain.models import ActionType, CorporateAction
from quant_trader.experiments.runner import (
    run_cost_sensitivity,
    run_parameter_sensitivity,
    run_variant,
)
from quant_trader.reporting.reports import write_report


def make_config() -> AppConfig:
    return AppConfig(
        strategy=StrategyConfig(
            sma_fast=20,
            sma_slow=50,
            min_history=60,
            min_avg_daily_value_eur=1_000,
            max_positions=3,
        ),
        risk=RiskConfig(starting_cash_eur=5_000, max_position_value_eur=1_000),
        execution=ExecutionConfig(),
        data={"database": "unused.duckdb"},
    )


def test_synthetic_strategy_backtest_and_report_are_reproducible(tmp_path) -> None:
    bars, reference = synthetic_market_data(date(2020, 1, 1), date(2022, 12, 31), 4)
    config = make_config()
    actions = [CorporateAction("SYN01", date(2021, 6, 1), ActionType.DIVIDEND, 1.0, "EUR")]

    first = run_variant(
        "E", bars, reference, config.strategy, config.risk, config.execution, actions
    )
    second = run_variant(
        "E", bars, reference, config.strategy, config.risk, config.execution, actions
    )
    cost_baseline = run_cost_sensitivity(bars, reference, config, (1.0,), actions).iloc[0]
    parameter_baseline = run_parameter_sensitivity(
        bars,
        reference,
        config,
        (config.strategy.neutral_zone,),
        (config.strategy.stop_atr_multiple,),
        actions,
    ).iloc[0]
    report_bars = pd.concat([bars, reference], ignore_index=True)
    report_path = write_report(
        first, config, report_bars, tmp_path / "run", data_source="synthetic test data"
    )
    report = pd.read_json(report_path)

    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    assert cost_baseline["total_return_net"] == first.metrics["total_return_net"]
    assert parameter_baseline["total_return_net"] == first.metrics["total_return_net"]
    assert not first.transactions.empty
    assert first.metadata["start_date"] == "2020-01-01"
    assert first.metadata["end_date"] == "2022-12-30"
    assert report.loc["strategy_version", "metadata"] == "0.1.0"
    assert report.loc["data_source", "metadata"] == "synthetic test data"
    assert report.loc["data_coverage", "metadata"]["instrument_count"] == 5
    assert (tmp_path / "run" / "report.md").exists()
    assert (tmp_path / "run" / "trades.csv").exists()
    assert (tmp_path / "run" / "equity_curve.html").exists()
    notebook = json.loads((tmp_path / "run" / "report.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell["metadata"].get("language") for cell in notebook["cells"])
    cell_ids = [cell["metadata"]["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    code_sources = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    for source in code_sources:
        compile(source, "report.ipynb", "exec")
    assert all(
        any(filename in source for source in code_sources)
        for filename in ("equity_curve.csv", "trades.csv", "transactions.csv")
    )
    assert (
        sum(
            "application/vnd.plotly.v1+json" in output.get("data", {})
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        )
        == 3
    )
    assert (
        sum(
            "text/html" in output.get("data", {})
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        )
        == 3
    )
