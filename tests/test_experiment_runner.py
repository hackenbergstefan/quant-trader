from types import SimpleNamespace

import pandas as pd

from pytrading import BrokerConfig, Portfolio
from pytrading.experiment_runner import (
    build_experiment_matrix,
    load_experiment_config,
    run_comparison,
    run_experiment_cases,
)
from pytrading.strategies.momentum import MomentumConfig, StrategyVariant


def _portfolio(close: pd.DataFrame, turnover: pd.DataFrame) -> Portfolio:
    universe = SimpleNamespace(
        close=lambda: close,
        turnover=lambda: turnover,
        market_metadata=dict,
    )
    return Portfolio(
        universe=universe,
        start_date=str(close.index[0].date()),
        end_date=str(close.index[-1].date()),
        broker=BrokerConfig(initial_cash=3000.0),
    )


def test_experiment_matrix_has_only_the_requested_variant_differences():
    specs = build_experiment_matrix()

    assert [spec.experiment_id for spec in specs] == [
        "ABC-W5",
        "ABC-M5",
        "ABC-M10",
        "AB-M10",
    ]
    assert [spec.variant for spec in specs] == [
        StrategyVariant.ABC,
        StrategyVariant.ABC,
        StrategyVariant.ABC,
        StrategyVariant.AB,
    ]
    allowed = {"topn", "rebalance_frequency", "position_sizing"}
    reference = vars(specs[0].config)
    for spec in specs[1:]:
        config = vars(spec.config)
        assert {key for key in reference if reference[key] != config[key]} <= allowed
    assert [
        (s.config.rebalance_frequency, s.config.topn, s.config.position_sizing)
        for s in specs
    ] == [
        ("weekly", 5, "inverse_volatility"),
        ("monthly", 5, "inverse_volatility"),
        ("monthly", 10, "inverse_volatility"),
        ("monthly", 10, "equal"),
    ]
    assert all(spec.config.max_position_weight == 0.2 for spec in specs)


def test_runner_archives_config_and_writes_csv_report_and_all_plots(
    monkeypatch, tmp_path
):
    from pytrading import experiment_runner

    dates = pd.bdate_range("2020-01-02", periods=4)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=dates)
    portfolio = _portfolio(
        close, pd.DataFrame(1000.0, index=dates, columns=close.columns)
    )
    specs = build_experiment_matrix()

    def fake_run_experiment_cases(_portfolio, cases):
        rows = []
        for index, (experiment_id, variant, _config) in enumerate(cases):
            net = pd.Series(
                [3000.0, 3010.0, 3020.0 + index, 3030.0 + index], index=dates
            )
            gross = net + 5.0
            rows.append(
                {
                    "Experiment ID": experiment_id,
                    "Variant": variant.value,
                    "Status": "success",
                    "Net Total Return": net.iloc[-1] / 3000.0 - 1,
                    "Gross Total Return": gross.iloc[-1] / 3000.0 - 1,
                    "Maximum Drawdown": -0.02 + index / 1000,
                    "Annualized Two-sided Turnover": 2.0 + index,
                    "Annualized Two-sided Turnover 2020": 2.0 + index,
                    "Annual Transaction Costs 2020": 10.0 + index,
                    "Sharpe": 0.5 + index / 10,
                    "_Net Equity": net,
                    "_Gross Equity": gross,
                }
            )
        return rows

    monkeypatch.setattr(
        experiment_runner, "run_experiment_cases", fake_run_experiment_cases
    )
    output_dir = run_comparison(portfolio, tmp_path, run_id="synthetic-run")

    expected_files = {
        "config.json",
        "results.csv",
        "equity_curves.csv",
        "report.html",
        "report.md",
        "equity_net.html",
        "equity_gross.html",
        "drawdowns.html",
        "annual_net_returns.html",
        "annual_turnover.html",
        "annual_transaction_costs.html",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    archived, restored_specs = load_experiment_config(output_dir / "config.json")
    assert archived["run_id"] == "synthetic-run"
    assert archived["broker"]["slippage"] == 0.0005
    assert [spec.experiment_id for spec in restored_specs] == [
        spec.experiment_id for spec in specs
    ]
    assert [spec.config for spec in restored_specs] == [spec.config for spec in specs]
    results = pd.read_csv(output_dir / "results.csv")
    assert results["Status"].eq("success").all()
    assert "ABC-W5 Net" in pd.read_csv(output_dir / "equity_curves.csv").columns
    assert "survivorship bias" in (output_dir / "report.md").read_text().lower()
    assert "percentage points" in (output_dir / "report.md").read_text().lower()


def test_runner_marks_invalid_variant_data_as_failure_not_success():
    dates = pd.bdate_range("2020-01-02", periods=3)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    empty_turnover = pd.DataFrame(index=dates)
    portfolio = _portfolio(close, empty_turnover)
    config = MomentumConfig(topn=1, min_avg_daily_turnover=0)

    rows = run_experiment_cases(portfolio, [("BROKEN", StrategyVariant.A, config)])

    assert rows[0]["Status"] == "error"
    assert "Keine historischen Umsatzdaten" in rows[0]["Error"]
    assert "Net Total Return" not in rows[0]
