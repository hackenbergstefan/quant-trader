from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pytrading import BrokerConfig, Portfolio
from pytrading.strategies.momentum import (
    MomentumConfig,
    StrategyVariant,
    _asset_execution_events,
    _atr_14,
    _backtest,
    _build_vectorbt_portfolio,
    _default_variant_config,
    _inverse_volatility_weights,
    _liquidity_eligibility,
    _market_regimes,
    _momentum_scores,
    _next_execution_date,
    _portfolio_weights,
    _realized_volatility,
    _trend_filter,
    backtest,
    rebalance_dates,
    signal_diagnostics,
    variant_experiment,
)


def _price_panel(periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    step = np.arange(periods)
    return pd.DataFrame(
        {
            "US": 100 * np.exp(0.0008 * step + 0.008 * np.sin(step / 5)),
            "EU": 90 * np.exp(0.0005 * step + 0.01 * np.sin(step / 7)),
            "OTHER": 70 * np.exp(0.0007 * step + 0.006 * np.cos(step / 9)),
        },
        index=dates,
    )


def _portfolio(close: pd.DataFrame, benchmarks=None) -> Portfolio:
    turnover = pd.DataFrame(1_000_000.0, index=close.index, columns=close.columns)
    market = {
        "US": {"country": "United States", "exchange": "NYQ"},
        "EU": {"country": "Germany", "exchange": "GER"},
        "OTHER": {"country": "Canada", "exchange": "TOR"},
    }
    universe = SimpleNamespace(
        close=lambda: close,
        turnover=lambda: turnover,
        market_metadata=lambda: market,
    )
    return Portfolio(
        universe,
        str(close.index[220].date()),
        str(close.index[-1].date()),
        BrokerConfig(),
        benchmarks or {},
    )


def test_momentum_requires_complete_history_and_handles_ties_and_nan():
    close = _price_panel(140)
    close["TIE"] = close["US"]
    close["MISSING"] = close["EU"]
    close.loc[close.index[-1], "MISSING"] = np.nan

    scores = _momentum_scores(close, MomentumConfig())

    assert scores.iloc[119].isna().all()
    assert scores.loc[close.index[-1], "US"] == scores.loc[close.index[-1], "TIE"]
    assert pd.isna(scores.loc[close.index[-1], "MISSING"])


def test_trend_eligibility_changes_cross_sectional_ranks_before_selection():
    dates = pd.bdate_range("2024-01-02", periods=140)
    step = np.arange(len(dates))
    close = pd.DataFrame(
        {
            "SLOW": 100 * np.exp(0.001 * step),
            "MID": 100 * np.exp(0.002 * step),
            "FAST": 100 * np.exp(0.003 * step),
        },
        index=dates,
    )
    eligibility = pd.DataFrame(True, index=dates, columns=close.columns)
    eligibility["FAST"] = False
    config = MomentumConfig(topn=1)

    scores_a = _momentum_scores(close, config)
    scores_b = _momentum_scores(close, config, eligibility=eligibility)

    assert scores_a.loc[dates[-1]].idxmax() == "FAST"
    assert scores_b.loc[dates[-1]].idxmax() == "MID"
    assert scores_a.loc[dates[-1], "MID"] != scores_b.loc[dates[-1], "MID"]


def test_legacy_config_is_monthly_but_ensemble_defaults_weekly(monkeypatch):
    assert MomentumConfig().rebalance_frequency == "monthly"
    assert _default_variant_config().rebalance_frequency == "weekly"
    assert _default_variant_config().topn == 5

    captured = {}
    monkeypatch.setattr(
        "pytrading.strategies.momentum.backtest",
        lambda portfolio, config, variant=None: captured.update(
            config=config, variant=variant
        ),
    )
    from pytrading.strategies.momentum import run_variant

    run_variant(None, "A")

    assert captured["config"].rebalance_frequency == "weekly"
    assert captured["config"].momentum_20_weight == pytest.approx(0.2)
    assert captured["config"].momentum_60_weight == pytest.approx(0.3)
    assert captured["config"].momentum_120_weight == pytest.approx(0.5)


def test_vectorbt_executes_at_next_open_and_marks_at_close():
    dates = pd.date_range("2024-01-02", periods=3)
    close = pd.DataFrame({"A": [100.0, 110.0, 120.0]}, index=dates)
    open_prices = pd.DataFrame({"A": [99.0, 105.0, 115.0]}, index=dates)
    weights = pd.DataFrame(np.nan, index=dates, columns=["A"])
    weights.loc[dates[1], "A"] = 1.0
    broker = BrokerConfig(initial_cash=1000, fees=0, fixed_fees=0, slippage=0)

    result = _build_vectorbt_portfolio(close, open_prices, weights, broker)

    orders = result.orders.records_readable
    assert orders.loc[0, "Timestamp"] == dates[1]
    assert orders.loc[0, "Price"] == pytest.approx(105.0)
    assert result.value().loc[dates[1]] == pytest.approx(1000 * 110 / 105)


def test_backtest_uses_open_after_signal_and_future_data_cannot_change_signal():
    close = _price_panel(320)
    dates = close.index
    open_prices = close * 0.99
    turnover = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    market = {
        column: {"country": "United States", "exchange": "NYQ"}
        for column in close.columns
    }
    universe = SimpleNamespace(
        close=lambda: close,
        open=lambda: open_prices,
        turnover=lambda: turnover,
        market_metadata=lambda: market,
    )
    portfolio = Portfolio(
        universe,
        str(dates[220].date()),
        str(dates[-1].date()),
        BrokerConfig(initial_cash=1000, fees=0, fixed_fees=0, slippage=0),
    )
    signal_date = dates[219]
    config = MomentumConfig(topn=2, min_avg_daily_turnover=0)
    original_scores = _momentum_scores(close, config)
    altered_close = close.copy()
    altered_close.loc[dates[220] :, :] *= 10
    altered_scores = _momentum_scores(altered_close, config)

    result = backtest(portfolio, config, variant="A")
    order = result.orders.records_readable.iloc[0]

    assert original_scores.loc[signal_date].equals(altered_scores.loc[signal_date])
    assert order["Timestamp"] > signal_date
    assert order["Price"] == pytest.approx(
        open_prices.loc[order["Timestamp"], order["Column"]]
    )


def test_missing_exchange_session_breaks_complete_momentum_lookback():
    dates = pd.bdate_range("2020-01-02", periods=230)
    close = pd.DataFrame({"US": np.linspace(100, 200, len(dates))}, index=dates)
    close.loc[dates[80], "US"] = np.nan
    metadata = {"US": {"exchange": "NYQ"}}

    scores = _momentum_scores(close, MomentumConfig(), metadata)

    assert pd.isna(scores.loc[dates[150], "US"])
    assert np.isfinite(scores.loc[dates[220], "US"])


def test_global_ranking_keeps_markets_open_on_different_holidays():
    dates = pd.bdate_range("2023-10-02", "2024-10-10")
    step = np.arange(len(dates))
    close = pd.DataFrame(
        {
            "US": 100 * np.exp(0.001 * step + 0.01 * np.sin(step / 5)),
            "EU": 90 * np.exp(0.001 * step + 0.01 * np.cos(step / 7)),
        },
        index=dates,
    )
    close.loc["2024-05-01", "EU"] = np.nan
    metadata = {
        "US": {"exchange": "NYQ"},
        "EU": {"exchange": "GER"},
    }

    scores = _momentum_scores(close, MomentumConfig(), metadata)

    assert scores.loc["2024-07-04"].notna().all()
    assert scores.loc["2024-05-01"].notna().all()


def test_trend_filter_requires_complete_averages_and_missing_close_is_false():
    close = _price_panel(240)
    close.loc[close.index[-1], "US"] = np.nan

    eligible = _trend_filter(close)

    assert not eligible.iloc[198].any()
    assert eligible.loc[close.index[-2], "EU"]
    assert not eligible.loc[close.index[-1], "US"]


def test_realized_volatility_uses_20_and_60_observed_returns():
    close = _price_panel(100)
    close.loc[close.index[30], "EU"] = np.nan

    volatility = _realized_volatility(close)

    assert pd.isna(volatility.iloc[59, 0])
    assert np.isfinite(volatility.iloc[60, 0])
    assert np.isfinite(volatility.iloc[-1, 1])
    assert volatility.iloc[-1, 0] > 0


def test_atr_14_is_available_as_a_separate_risk_metric():
    close = _price_panel(40)
    high = close + 1
    low = close - 1

    atr = _atr_14(high, low, close)

    assert atr.iloc[:13].isna().all().all()
    assert np.isfinite(atr.iloc[13]).all()
    assert atr.iloc[13, 0] == pytest.approx(2.0)


def test_inverse_volatility_weights_reapply_caps_and_keep_cash_when_needed():
    volatility = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=list("ABCDE"))

    capped = _inverse_volatility_weights(volatility, max_position_weight=0.3)
    fewer = _inverse_volatility_weights(volatility.iloc[:3], max_position_weight=0.2)

    assert capped.max() == pytest.approx(0.3)
    assert capped.sum() == pytest.approx(1.0)
    assert capped.is_monotonic_decreasing
    assert fewer.sum() == pytest.approx(0.6)
    assert fewer.max() == pytest.approx(0.2)


def test_liquidity_filter_preserves_75_percent_observation_threshold():
    dates = pd.bdate_range("2024-01-02", periods=30)
    close = pd.DataFrame(100.0, index=dates, columns=["PASS", "FAIL"])
    turnover = pd.DataFrame(100.0, index=dates, columns=close.columns)
    turnover.loc[dates[10:15], "PASS"] = np.nan
    turnover.loc[dates[10:16], "FAIL"] = np.nan
    config = MomentumConfig(liquidity_lookback=20, min_avg_daily_turnover=50)

    eligible = _liquidity_eligibility(close, turnover, config)

    assert eligible.loc[dates[-1], "PASS"]
    assert not eligible.loc[dates[-1], "FAIL"]


def test_market_regime_transitions_and_unknown_history():
    values = np.r_[
        np.full(150, 100.0),
        np.linspace(101, 200, 50),
        np.full(49, 100.0),
        151.0,
        90.0,
    ]
    dates = pd.bdate_range("2020-01-02", periods=len(values))
    benchmark = pd.Series(values, index=dates)
    regimes = _market_regimes({"USA": benchmark}, dates)

    assert regimes.loc[dates[199], "USA"] == "bullish"
    assert regimes.loc[dates[249], "USA"] == "neutral"
    assert regimes.loc[dates[250], "USA"] == "bearish"
    assert regimes["EUROPE"].eq("unknown").all()
    stale = _market_regimes({"USA": benchmark.iloc[:250]}, dates)
    assert stale.loc[dates[250] :, "USA"].eq("unknown").all()

    gap_dates = pd.bdate_range("2020-01-02", periods=450)
    gap_values = pd.Series(np.linspace(100, 300, len(gap_dates)), index=gap_dates)
    gap_values.iloc[220] = np.nan
    gap_regimes = _market_regimes({"USA": gap_values}, gap_dates)
    assert gap_regimes.loc[gap_dates[220], "USA"] == "unknown"
    assert gap_regimes.loc[gap_dates[419], "USA"] == "unknown"
    assert gap_regimes.loc[gap_dates[420], "USA"] == "bullish"


def test_weekly_monthly_rebalance_dates_use_last_available_session():
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-02-01"]
    )
    prices = pd.DataFrame({"A": range(5)}, index=dates)

    assert list(rebalance_dates(prices, "weekly")) == [dates[2], dates[3], dates[4]]
    assert list(rebalance_dates(prices, "monthly")) == [dates[3], dates[4]]


@pytest.mark.parametrize(
    ("signal_date", "execution_date", "calendar"),
    [
        ("2024-07-03", "2024-07-05", "XNYS"),
        ("2024-10-02", "2024-10-04", "XETR"),
    ],
)
def test_next_session_execution_respects_us_and_european_holidays(
    signal_date, execution_date, calendar
):
    dates = pd.to_datetime([signal_date, execution_date])
    close = pd.DataFrame({"ISIN": [100.0, 101.0]}, index=dates)

    scheduled = _next_execution_date(close, "ISIN", dates[0], calendar)

    assert scheduled == dates[1]
    assert scheduled > dates[0]


def test_asset_execution_events_use_only_post_signal_sessions():
    dates = pd.to_datetime(["2024-07-03", "2024-07-05"])
    close = pd.DataFrame({"US": [100.0, 101.0]}, index=dates)
    config = MomentumConfig(rebalance_frequency="weekly")

    events = _asset_execution_events(
        close, close.loc[dates[1] :], config, {"US": {"exchange": "NYQ"}}
    )

    assert events[(dates[1], dates[0])] == ["US"]


def test_neutral_regime_caps_regional_exposure_without_refilling():
    benchmark_values = np.r_[
        np.full(150, 100.0),
        np.linspace(101, 200, 50),
        np.full(49, 100.0),
        151.0,
    ]
    benchmark_dates = pd.bdate_range("2020-01-02", periods=len(benchmark_values))
    signal_date = benchmark_dates[-1]
    execution_date = signal_date + pd.offsets.BDay(1)
    close = pd.DataFrame(
        {"USA": [100.0, 101.0], "EUROPE": [100.0, 101.0]},
        index=[signal_date, execution_date],
    )
    prices = close.loc[[execution_date]]
    scores = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    config = MomentumConfig(
        topn=2,
        rebalance_frequency="weekly",
        neutral_exposure=0.25,
    )
    metadata = {
        "USA": {"country": "United States"},
        "EUROPE": {"country": "Germany"},
    }
    benchmarks = {
        "USA": pd.Series(benchmark_values, index=benchmark_dates),
        "EUROPE": pd.Series(benchmark_values, index=benchmark_dates),
    }

    weights = _portfolio_weights(
        close,
        prices,
        scores,
        None,
        config,
        signals=frozenset("D"),
        metadata=metadata,
        benchmarks=benchmarks,
    )

    assert weights.loc[execution_date].sum() == pytest.approx(0.5)
    assert weights.loc[execution_date, "USA"] == pytest.approx(0.25)
    assert weights.loc[execution_date, "EUROPE"] == pytest.approx(0.25)


def test_asynchronous_region_rotation_stages_buys_after_sells():
    import exchange_calendars

    dates = pd.bdate_range("2024-04-29", "2024-05-07")
    europe_calendar = exchange_calendars.get_calendar("XETR")
    us_calendar = exchange_calendars.get_calendar("XNYS")
    close = pd.DataFrame(index=dates, columns=["EUROPE", "USA"], dtype=float)
    open_prices = close.copy()
    for column, calendar in (
        ("EUROPE", europe_calendar),
        ("USA", us_calendar),
    ):
        for date in dates:
            if calendar.is_session(date):
                close.loc[date, column] = 100.0
                open_prices.loc[date, column] = 101.0

    signal_date = pd.Timestamp("2024-04-30")
    prices = close.loc[close.index > signal_date]
    scores = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    scores.loc[signal_date] = 1.0
    volatility = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    volatility.loc[signal_date] = [4.0, 1.0]
    benchmark_dates = pd.bdate_range("2023-01-02", signal_date)
    benchmark = pd.Series(
        np.linspace(100, 200, len(benchmark_dates)), index=benchmark_dates
    )
    metadata = {
        "EUROPE": {"country": "Germany", "exchange": "GER"},
        "USA": {"country": "United States", "exchange": "NYQ"},
    }
    initial = pd.Series({"EUROPE": 0.8, "USA": 0.2})
    config = MomentumConfig(
        topn=2,
        position_sizing="inverse_volatility",
        max_position_weight=1.0,
    )

    weights = _portfolio_weights(
        close,
        prices,
        scores,
        None,
        config,
        signals=frozenset("CD"),
        metadata=metadata,
        benchmarks={"EUROPE": benchmark, "USA": benchmark},
        volatility=volatility,
        execution_prices=open_prices,
        initial_weights=initial,
    )

    assert pd.isna(weights.loc["2024-05-01", "USA"])
    assert weights.loc["2024-05-02", "EUROPE"] == pytest.approx(0.2)
    assert weights.loc["2024-05-03", "USA"] == pytest.approx(0.8)
    held = initial.copy()
    for date in prices.index:
        orders = weights.loc[date].dropna()
        held.loc[orders.index] = orders
        assert held.sum() <= 1.0 + 1e-12
    assert held.to_dict() == pytest.approx({"EUROPE": 0.2, "USA": 0.8})


def test_all_variants_run_identically_scheduled_vectorbt_engine():
    close = _price_panel()
    benchmark = pd.Series(np.linspace(80, 150, len(close)), index=close.index)
    portfolio = _portfolio(close, {"USA": benchmark, "EUROPE": benchmark})
    config = MomentumConfig(topn=2, min_avg_daily_turnover=0)

    report = variant_experiment(portfolio, config)

    assert list(report["Variant"]) == [variant.value for variant in StrategyVariant]
    assert report["Start Date"].nunique() == 1
    assert report["End Date"].nunique() == 1
    assert {
        "CAGR",
        "Total Return",
        "Maximum Drawdown",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Profit Factor",
        "Number of Trades",
        "Average Holding Period (sessions)",
        "Average Exposure",
        "Maximum Exposure",
        "Cash Allocation",
        "Turnover",
        "Transaction Costs",
        "Average EUROPE Exposure",
        "Maximum USA Exposure",
    }.issubset(report.columns)
    pd.testing.assert_frame_equal(report, variant_experiment(portfolio, config))


def test_diagnostics_expose_signal_values_and_missing_regime_reason():
    close = _price_panel()
    portfolio = _portfolio(close)
    diagnostics = signal_diagnostics(
        portfolio,
        "ABCD",
        MomentumConfig(topn=2, min_avg_daily_turnover=0),
    )

    assert {
        "momentum_score",
        "momentum_rank",
        "trend_eligible",
        "realized_volatility",
        "position_sizing_weight",
        "regional_regime",
        "regional_exposure_limit",
        "exclusion_reason",
    }.issubset(diagnostics.columns)
    assert (
        diagnostics.loc[diagnostics["region"] == "USA", "regional_regime"]
        .eq("unknown")
        .all()
    )
    assert diagnostics["exclusion_reason"].str.contains("unknown_regional_regime").any()


def test_backtest_deduplicates_instruments_by_canonical_column():
    close = _price_panel().iloc[:, :2]
    close.columns = ["US-ISIN", "EU-ISIN"]
    duplicate = pd.concat([close, close.iloc[:, [0]]], axis=1)
    duplicate.columns = ["US-ISIN", "EU-ISIN", "US-ISIN"]

    result = _backtest(
        duplicate,
        None,
        BrokerConfig(),
        str(close.index[220].date()),
        str(close.index[-1].date()),
        MomentumConfig(topn=2),
        variant="A",
    )

    assert result.close.shape[1] == 2
