import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pytrading import BrokerConfig, Portfolio
from pytrading.rolling_analysis import (
    _null_metric_reasons,
    _pairwise_comparisons,
    _window_dependence_note,
    aggregate_rolling_results,
    build_rolling_windows,
    refresh_rolling_exports,
    run_rolling_comparison,
)
from pytrading.strategies.momentum import MomentumConfig, _prepare_backtest_inputs


def _portfolio(close: pd.DataFrame) -> Portfolio:
    universe = SimpleNamespace(
        close=lambda: close,
        turnover=lambda: pd.DataFrame(
            100_000.0, index=close.index, columns=close.columns
        ),
        market_metadata=dict,
    )
    return Portfolio(
        universe=universe,
        start_date="2020-01-01",
        end_date=str(close.index[-1].date()),
        broker=BrokerConfig(initial_cash=3000.0),
    )


def test_rolling_windows_use_exclusive_ends_and_separate_incomplete_periods():
    windows = build_rolling_windows("2010-01-01", "2026-11-06")

    assert len(windows) == 17
    assert sum(window.complete for window in windows) == 14
    last_full = next(window for window in windows if window.start_date.year == 2023)
    assert last_full.end_date_exclusive == pd.Timestamp("2026-01-01")
    assert last_full.observed_end_exclusive == pd.Timestamp("2026-01-01")
    assert last_full.complete
    assert [window.start_date.year for window in windows if not window.complete] == [
        2024,
        2025,
        2026,
    ]
    assert windows[-1].end_date_exclusive == pd.Timestamp("2029-01-01")
    assert windows[-1].observed_end_exclusive == pd.Timestamp("2026-11-07")
    assert windows[-1].backtest_end_date == "2026-11-06"

    non_overlapping = build_rolling_windows("2010-01-01", "2026-11-06", step_years=3)
    assert [window.start_date.year for window in non_overlapping] == [
        2010,
        2013,
        2016,
        2019,
        2022,
        2025,
    ]
    assert sum(window.complete for window in non_overlapping) == 5
    assert not non_overlapping[-1].complete
    assert "do not overlap" in _window_dependence_note(3, 3)
    assert "overlap" in _window_dependence_note(3, 1)


def test_indicator_history_precedes_window_but_simulation_slice_does_not():
    dates = pd.bdate_range("2019-01-01", periods=330)
    close = pd.DataFrame(
        {
            "A": 10 * (1.002 ** np.arange(len(dates))),
            "B": 20 * (1.001 ** np.arange(len(dates))),
        },
        index=dates,
    )
    turnover = pd.DataFrame(100_000.0, index=dates, columns=close.columns)
    start = dates[250].date().isoformat()
    exclusive_end = dates[280]
    backtest_end = (exclusive_end - pd.Timedelta(days=1)).date().isoformat()
    config = MomentumConfig(topn=2, min_avg_daily_turnover=0)

    prices, _, _, full_volatility = _prepare_backtest_inputs(
        close, turnover, start, backtest_end, config
    )
    _, _, _, window_only_volatility = _prepare_backtest_inputs(
        close.loc[start:], turnover.loc[start:], start, backtest_end, config
    )

    assert prices.index.min() >= pd.Timestamp(start)
    assert prices.index.max() < exclusive_end
    assert pd.notna(full_volatility.loc[start, "A"])
    assert pd.isna(window_only_volatility.loc[start, "A"])


def test_rolling_summary_excludes_partial_windows_and_ignores_missing_metrics():
    details = pd.DataFrame(
        [
            {
                "Experiment ID": "AB-M5",
                "Window ID": "w1",
                "Complete Window": True,
                "Window Status": "complete",
                "Status": "success",
                "Net Total Return": 0.1,
                "Net CAGR": 0.08,
                "Sharpe Ratio": 0.5,
            },
            {
                "Experiment ID": "AB-M5",
                "Window ID": "w2",
                "Complete Window": True,
                "Window Status": "complete",
                "Status": "success",
                "Net Total Return": -0.1,
                "Net CAGR": -0.08,
                "Sharpe Ratio": None,
            },
            {
                "Experiment ID": "AB-M5",
                "Window ID": "w3",
                "Complete Window": False,
                "Window Status": "incomplete",
                "Status": "success",
                "Net Total Return": 1.0,
                "Net CAGR": 1.0,
                "Sharpe Ratio": 9.0,
            },
        ]
    )

    summary = aggregate_rolling_results(details).set_index("Metric")

    assert summary.loc["Net CAGR", "Number of Metric Observations"] == 2
    assert summary.loc["Net CAGR", "Arithmetic Mean"] == 0
    assert summary.loc["Net CAGR", "Worst CAGR"] == -0.08
    assert summary.loc["Net CAGR", "Best CAGR"] == 0.08
    assert summary.loc["Net CAGR", "Number of Positive Return Windows"] == 1
    assert summary.loc["Sharpe Ratio", "Number of Metric Observations"] == 1
    assert summary.loc["Sharpe Ratio", "Arithmetic Mean"] == 0.5


def test_pairwise_comparison_reports_deltas_and_excludes_partial_aggregates():
    rows = []
    values = {
        "w1": {
            "AB-M5": (0.10, 1.0, -0.10, 100.0),
            "ABC-M5": (0.20, 0.9, -0.05, 130.0),
            "AB-M10": (0.08, 0.8, -0.12, 150.0),
            "ABC-M10": (0.17, 0.85, -0.06, 170.0),
        },
        "w2": {
            "AB-M5": (0.40, 1.5, -0.20, 200.0),
            "ABC-M5": (0.20, 0.9, -0.05, 130.0),
            "AB-M10": (0.08, 0.8, -0.12, 150.0),
            "ABC-M10": (0.17, 0.85, -0.06, 170.0),
        },
    }
    for window_id, variants in values.items():
        for variant, (cagr, sharpe, drawdown, costs) in variants.items():
            rows.append(
                {
                    "Window ID": window_id,
                    "Experiment ID": variant,
                    "Start Date": "2020-01-01",
                    "End Date": "2023-01-01",
                    "Window Status": "complete" if window_id == "w1" else "incomplete",
                    "Complete Window": window_id == "w1",
                    "Status": "success",
                    "Net CAGR": cagr,
                    "Sharpe Ratio": sharpe,
                    "Maximum Drawdown": drawdown,
                    "Transaction Costs": costs,
                    "Transaction Costs / Initial Capital": costs / 3000.0,
                }
            )

    comparisons = _pairwise_comparisons(pd.DataFrame(rows))
    pair = comparisons.loc[
        comparisons["Record Type"].eq("window")
        & comparisons["Reference Variant A"].eq("AB-M5")
        & comparisons["Compared Variant B"].eq("ABC-M5")
        & comparisons["Window ID"].eq("w1")
    ].iloc[0]
    aggregate = comparisons.loc[
        comparisons["Record Type"].eq("aggregate")
        & comparisons["Reference Variant A"].eq("AB-M5")
        & comparisons["Compared Variant B"].eq("ABC-M5")
    ].iloc[0]

    assert pair["CAGR Difference (pp) (B-A)"] == 10.0
    assert np.isclose(pair["Sharpe Difference (B-A)"], -0.1)
    assert np.isclose(pair["Maximum Drawdown Difference (pp) (B-A)"], 5.0)
    assert pair["Transaction Costs Difference (EUR) (B-A)"] == 30.0
    assert pair["A Better: Net CAGR"] == np.False_
    assert pair["A Better: Sharpe Ratio"]
    assert not pair["A Better: Maximum Drawdown"]
    assert pair["A Better: Transaction Costs"]
    assert aggregate["Mean CAGR Difference (pp) (B-A)"] == 10.0
    assert aggregate["A Better Share: Net CAGR"] == 0.0
    assert aggregate["B Better Share: Net CAGR"] == 1.0
    assert aggregate["Number of Complete Windows"] == 1


def test_null_metrics_include_reasons_for_missing_years_and_undefined_metrics():
    reasons = json.loads(
        _null_metric_reasons(
            {
                "Status": "success",
                "Start Date": "2020-01-01",
                "Observed End Date": "2022-12-30",
                "Annualized Two-sided Turnover 2019": None,
                "Average Holding Period (sessions)": None,
                "Benchmarks Available": None,
            }
        )
    )

    assert "No valuation sessions" in reasons["Annualized Two-sided Turnover 2019"]
    assert reasons["Average Holding Period (sessions)"] == (
        "No FIFO buy lot was fully closed."
    )
    assert reasons["Benchmarks Available"] == (
        "No benchmark series were configured or available."
    )


def test_rolling_run_exports_consistent_rows_and_resets_each_window(
    monkeypatch, tmp_path
):
    from pytrading import rolling_analysis

    dates = pd.bdate_range("2020-01-01", "2023-01-03")
    close = pd.DataFrame({"A": np.linspace(10.0, 20.0, len(dates))}, index=dates)
    portfolio = _portfolio(close)
    calls = []

    def fake_run_experiment_cases(window_portfolio, cases):
        calls.append((window_portfolio, tuple(cases)))
        dates_in_window = close.loc[
            window_portfolio.start_date : window_portfolio.end_date
        ].index
        rows = []
        for index, (experiment_id, variant, _config) in enumerate(cases):
            equity = pd.Series(
                np.linspace(3000.0, 3300.0 + index * 10, len(dates_in_window)),
                index=dates_in_window,
            )
            rows.append(
                {
                    "Experiment ID": experiment_id,
                    "Variant": variant.value,
                    "Status": "success",
                    "Initial Capital": 3000.0,
                    "Final Portfolio Value": float(equity.iloc[-1]),
                    "Gross Final Portfolio Value": float(equity.iloc[-1] + 10),
                    "Net Total Return": float(equity.iloc[-1] / 3000.0 - 1),
                    "Net CAGR": 0.08 + index / 100,
                    "Gross Total Return": float((equity.iloc[-1] + 10) / 3000.0 - 1),
                    "Gross CAGR": 0.09 + index / 100,
                    "Maximum Drawdown": -0.1 + index / 100,
                    "Sharpe": 0.5 + index / 10,
                    "Sortino": 0.3 + index / 10,
                    "Calmar": 0.8 + index / 10,
                    "Average Exposure": 0.7,
                    "Cash Allocation": 0.3,
                    "Transaction Costs": 20.0 + index,
                    "Proportional Fees Costs": 10.0,
                    "Fixed Fees Costs": 5.0,
                    "Transaction Fees": 15.0,
                    "Slippage Costs": 5.0,
                    "Negative Months": 1,
                    "Positive Month Share": 0.5,
                    "Average Covariance-Based Portfolio Volatility": 0.2,
                    "Weight-Weighted Average Constituent Volatility": 0.25,
                    "_Net Equity": equity,
                    "_Gross Equity": equity + 10,
                }
            )
            if window_portfolio.start_date == "2020-01-01":
                rows[-1]["Annualized Two-sided Turnover 2019"] = None
        return rows

    monkeypatch.setattr(
        rolling_analysis, "run_experiment_cases", fake_run_experiment_cases
    )
    monkeypatch.setattr(
        rolling_analysis, "_git_revision", lambda _path: "test-revision"
    )

    output_dir = run_rolling_comparison(portfolio, tmp_path)

    assert len(calls) == 4
    assert len({id(call[0]) for call in calls}) == 4
    assert all(call[0].broker.initial_cash == 3000.0 for call in calls)
    assert all(
        [case[0] for case in call[1]] == ["AB-M5", "ABC-M5", "AB-M10", "ABC-M10"]
        for call in calls
    )
    detail_csv = pd.read_csv(output_dir / "rolling_3y_results.csv")
    detail_json = json.loads(
        (output_dir / "rolling_3y_results.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (output_dir / "rolling_3y_metadata.json").read_text(encoding="utf-8")
    )
    assert list(detail_json[0]) == list(detail_csv.columns)
    assert metadata["detail_fields"] == list(detail_csv.columns)
    assert metadata["git_revision"] == "test-revision"
    assert detail_csv["Status"].eq("success").all()
    assert detail_csv["Complete Window"].sum() == 4
    assert detail_csv["Initial Capital"].eq(3000.0).all()
    assert detail_csv.loc[:3, "End Date"].eq("2023-01-01").all()
    assert detail_csv.loc[:3, "Observed End Date"].eq("2022-12-30").all()
    assert detail_csv.loc[4:, "Observed End Date"].eq("2023-01-03").all()
    assert detail_csv["Positive Months"].gt(0).all()
    reasons = json.loads(detail_json[0]["Null Metric Reasons"])
    assert "Average Holding Period (sessions)" in reasons
    assert "Annualized Two-sided Turnover 2019" in reasons
    next_window_reasons = json.loads(detail_json[4]["Null Metric Reasons"])
    assert "Annualized Two-sided Turnover 2019" in next_window_reasons
    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Each window starts with fresh initial capital" in report
    assert "AB-M5" in report
    refresh_rolling_exports(output_dir)
    refreshed = pd.read_csv(output_dir / "rolling_3y_results.csv")
    assert "Null Metric Reasons" in refreshed.columns


def test_standard_runner_passes_requested_step_years(monkeypatch, tmp_path):
    from pytrading import data_helper, rolling_analysis

    dates = pd.bdate_range("2020-01-02", periods=3)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    universe = _portfolio(close).universe
    captured = {}

    monkeypatch.setattr(data_helper, "Universe", lambda: universe)

    def fake_run_rolling_comparison(portfolio, output_root, **kwargs):
        captured["portfolio"] = portfolio
        captured["output_root"] = output_root
        captured.update(kwargs)
        return output_root

    monkeypatch.setattr(
        rolling_analysis, "run_rolling_comparison", fake_run_rolling_comparison
    )

    output = rolling_analysis.run_standard_rolling_comparison(
        start_date="2020-01-01",
        output_root=tmp_path,
        step_years=3,
    )

    assert output == tmp_path
    assert captured["step_years"] == 3
    assert captured["portfolio"].end_date == "2020-01-06"
