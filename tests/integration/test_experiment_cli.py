from datetime import date

from typer.testing import CliRunner

from quant_trader import cli
from quant_trader.config import AppConfig, ExecutionConfig, RiskConfig, StrategyConfig
from quant_trader.data.synthetic import synthetic_market_data


def test_experiment_run_report_generates_comparison_and_variant_reports(
    monkeypatch, tmp_path
) -> None:
    bars, reference = synthetic_market_data(date(2020, 1, 1), date(2021, 12, 31), 4)
    config = AppConfig(
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
    monkeypatch.setattr(cli, "_read_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "_load_data",
        lambda *_args, **_kwargs: (bars, reference, [], "synthetic test data"),
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "experiment",
            "run",
            "--report",
            "--data-source",
            "synthetic",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "B_simple_momentum_equity_curve.csv").exists()
    assert (tmp_path / "reports" / "B_simple_momentum" / "report.json").exists()
    assert (tmp_path / "reports" / "A_euro_stoxx_50_buy_and_hold" / "equity_curve.html").exists()
    assert len(list((tmp_path / "reports").glob("*/report.ipynb"))) == 6

    regenerated = CliRunner().invoke(
        cli.app,
        [
            "report",
            "generate",
            "--run-dir",
            str(tmp_path / "reports" / "B_simple_momentum"),
        ],
    )
    assert regenerated.exit_code == 0, regenerated.output
    assert "report.ipynb" in regenerated.output
