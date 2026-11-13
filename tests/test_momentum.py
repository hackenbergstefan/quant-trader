import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pytrading import BrokerConfig, Portfolio
from pytrading.strategies.momentum import (
    VARIANT_SIGNALS,
    MomentumConfig,
    StrategyVariant,
    _asset_execution_events,
    _atr_14,
    _backtest,
    _build_vectorbt_portfolio,
    _default_variant_config,
    _holding_period_metrics,
    _inverse_volatility_weights,
    _liquidity_eligibility,
    _market_regimes,
    _momentum_scores,
    _next_execution_date,
    _portfolio_weights,
    _position_weight_metrics,
    _realized_volatility,
    _trend_filter,
    _turnover_metrics,
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


class _TestUniverse:
    def __init__(self, close, turnover, market):
        self._close = close
        self._turnover = turnover
        self._market = market

    def close(self):
        return self._close

    def turnover(self):
        return self._turnover

    def market_metadata(self):
        return self._market

    def __getstate__(self):
        raise AssertionError("Universe objects must not cross the process boundary")


def _portfolio(close: pd.DataFrame, benchmarks=None) -> Portfolio:
    turnover = pd.DataFrame(1_000_000.0, index=close.index, columns=close.columns)
    market = {
        "US": {"country": "United States", "exchange": "NYQ"},
        "EU": {"country": "Germany", "exchange": "GER"},
        "OTHER": {"country": "Canada", "exchange": "TOR"},
    }
    universe = _TestUniverse(close, turnover, market)
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


def test_backtest_aligns_open_columns_after_dropping_empty_period_columns():
    close = _price_panel(320)
    dates = close.index
    start_date = str(dates[220].date())
    end_date = str(dates[-1].date())
    close.loc[start_date:end_date, "OTHER"] = np.nan
    open_prices = close * 0.99

    result = _backtest(
        close,
        None,
        BrokerConfig(),
        start_date,
        end_date,
        MomentumConfig(topn=2, min_avg_daily_turnover=0),
        open_prices=open_prices,
        variant="A",
    )

    assert "OTHER" not in result.wrapper.columns
    assert result.wrapper.columns.equals(close.columns[:2])


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


def test_position_weight_metrics_use_actual_positions_and_annualized_volatility():
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame({"A": [100.0] * 3, "B": [50.0] * 3}, index=dates)
    assets = pd.DataFrame({"A": [1.0, 2.0, 0.0], "B": [2.0, 1.0, 0.0]}, index=dates)
    values = pd.Series(1000.0, index=dates)
    volatility = pd.DataFrame({"A": [0.2, 0.4, 0.4], "B": [0.4, 0.2, 0.3]}, index=dates)
    result = SimpleNamespace(value=lambda: values, assets=lambda: assets)

    metrics, average_weight, maximum_weight = _position_weight_metrics(
        result, close, volatility, max_position_weight=0.2
    )

    assert metrics["Average Position Weight"] == pytest.approx(0.1125)
    assert metrics["Median Position Weight"] == pytest.approx(0.1)
    assert metrics["Largest Position Weight"] == pytest.approx(0.2)
    assert metrics["Average Position Volatility"] == pytest.approx(0.3)
    assert metrics["Weighted Average Portfolio Volatility"] == pytest.approx(0.33)
    assert metrics["Weight Concentration HHI"] == pytest.approx(0.59)
    assert metrics["Time at or Above Position Cap"] == pytest.approx(1 / 3)
    assert average_weight.tolist() == pytest.approx([0.1, 0.125, 0.0])
    assert maximum_weight.tolist() == pytest.approx([0.1, 0.2, 0.0])


def test_signal_c_alone_controls_inverse_volatility_weighting(monkeypatch):
    assert VARIANT_SIGNALS == {
        StrategyVariant.A: frozenset("A"),
        StrategyVariant.AB: frozenset("AB"),
        StrategyVariant.ABC: frozenset("ABC"),
        StrategyVariant.ABD: frozenset("ABD"),
        StrategyVariant.ABCD: frozenset("ABCD"),
    }
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame(100.0, index=dates, columns=list("ABC"))
    scores = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    scores.loc[dates[0]] = 1.0
    volatility = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    volatility.loc[dates[0]] = [1.0, 2.0, 4.0]
    monkeypatch.setattr(
        "pytrading.strategies.momentum._asset_execution_events",
        lambda *_args, **_kwargs: {(dates[1], dates[0]): list(close.columns)},
    )
    config = MomentumConfig(topn=3, max_position_weight=1.0)

    equal_weights = _portfolio_weights(
        close,
        close,
        scores,
        None,
        config,
        signals=VARIANT_SIGNALS[StrategyVariant.AB],
    )
    inverse_weights = _portfolio_weights(
        close,
        close,
        scores,
        None,
        config,
        signals=VARIANT_SIGNALS[StrategyVariant.ABC],
        volatility=volatility,
    )

    assert equal_weights.loc[dates[1]].to_dict() == pytest.approx(
        {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    )
    assert inverse_weights.loc[dates[1]].to_dict() == pytest.approx(
        {"A": 4 / 7, "B": 2 / 7, "C": 1 / 7}
    )
    with pytest.raises(ValueError, match="Signal C benötigt Volatilitätsdaten"):
        _portfolio_weights(
            close,
            close,
            scores,
            None,
            config,
            signals=VARIANT_SIGNALS[StrategyVariant.ABC],
        )


def test_equal_weight_topn_respects_position_cap_and_keeps_remaining_cash(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame(100.0, index=dates, columns=list("ABCDE"))
    scores = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    scores.loc[dates[0]] = [5.0, 4.0, 3.0, 2.0, 1.0]
    monkeypatch.setattr(
        "pytrading.strategies.momentum._asset_execution_events",
        lambda *_args, **_kwargs: {(dates[1], dates[0]): list(close.columns)},
    )

    weights = _portfolio_weights(
        close,
        close,
        scores,
        None,
        MomentumConfig(topn=3, max_position_weight=0.2),
        signals=VARIANT_SIGNALS[StrategyVariant.AB],
    )

    target = weights.loc[dates[1]].fillna(0)
    assert target[target.gt(0)].to_dict() == pytest.approx(
        {"A": 0.2, "B": 0.2, "C": 0.2}
    )
    assert target.sum() == pytest.approx(0.6)
    assert target.max() <= 0.2


def test_experiment_config_matches_requested_period_and_broker_setup():
    experiment = runpy.run_path(str(Path(__file__).parents[1] / "experiment.py"))
    config = experiment["_experiment_config"]()

    assert experiment["variant_experiment"] is variant_experiment
    assert config.topn == 5
    assert config.rebalance_frequency == "monthly"
    assert experiment["START_DATE"] == "2010-01-01"
    assert experiment["END_DATE"] == "2017-01-01"
    assert experiment["_experiment_broker_config"]() == BrokerConfig(
        3000.0, 0.001, 1.0, 0.0005
    )


def test_turnover_metrics_full_five_position_replacement_and_annualization():
    dates = pd.bdate_range("2024-01-02", periods=126)
    values = pd.Series(1000.0, index=dates)
    orders = []
    order_id = 0
    for instrument in "ABCDE":
        orders.append((order_id, instrument, dates[0], 10.0, 20.0, 0.0, "Buy"))
        order_id += 1
    for instrument in "ABCDE":
        orders.append((order_id, instrument, dates[63], 10.0, 20.0, 0.0, "Sell"))
        order_id += 1
    for instrument in "FGHIJ":
        orders.append((order_id, instrument, dates[63], 10.0, 20.0, 0.0, "Buy"))
        order_id += 1
    records = pd.DataFrame(
        orders,
        columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"],
    )

    metrics = _turnover_metrics(records, values)

    assert metrics["One-sided Turnover"] == pytest.approx(2.0)
    assert metrics["Two-sided Turnover"] == pytest.approx(3.0)
    assert metrics["Annualized One-sided Turnover"] == pytest.approx(4.0)
    assert metrics["Annualized Two-sided Turnover"] == pytest.approx(6.0)
    assert metrics["Annualized One-sided Turnover 2024"] == pytest.approx(4.0)
    assert metrics["Annualized Two-sided Turnover 2024"] == pytest.approx(6.0)


def test_execution_costs_separate_recorded_fees_and_slippage():
    from pytrading.strategies.momentum import _execution_cost_metrics

    dates = pd.bdate_range("2024-01-02", periods=2)
    reference_prices = pd.DataFrame({"A": [10.0, 10.0]}, index=dates)
    orders = pd.DataFrame(
        [
            (0, "A", dates[0], 10.0, 10.005, 2.0, "Buy"),
            (1, "A", dates[1], 10.0, 9.995, 2.0, "Sell"),
        ],
        columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"],
    )

    fees, slippage = _execution_cost_metrics(orders, reference_prices)

    assert fees == pytest.approx(4.0)
    assert slippage == pytest.approx(0.1)


def test_holding_period_for_single_buy_and_sell_uses_execution_dates():
    dates = pd.bdate_range("2024-01-02", periods=5)
    close = pd.DataFrame({"A": [10.0] * len(dates)}, index=dates)
    orders = pd.DataFrame(
        [
            (0, "A", dates[0], 10.0, 10.0, 0.0, "Buy"),
            (1, "A", dates[2], 10.0, 10.0, 0.0, "Sell"),
        ],
        columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"],
    )

    metrics = _holding_period_metrics(orders, close)

    assert metrics["Completed Trade Lots"] == 1
    assert metrics["Average Holding Period (sessions)"] == 2
    assert metrics["Median Holding Period (sessions)"] == 2
    assert metrics["Completed Positions"] == 1
    assert metrics["Average Position Duration (sessions)"] == 2


def test_holding_period_fifo_handles_additions_and_partial_sales():
    dates = pd.bdate_range("2024-01-02", periods=5)
    close = pd.DataFrame({"A": [10.0] * len(dates)}, index=dates)
    orders = pd.DataFrame(
        [
            (0, "A", dates[0], 10.0, 10.0, 0.0, "Buy"),
            (1, "A", dates[1], 5.0, 10.0, 0.0, "Buy"),
            (2, "A", dates[2], 8.0, 10.0, 0.0, "Sell"),
            (3, "A", dates[4], 7.0, 10.0, 0.0, "Sell"),
        ],
        columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"],
    )

    metrics = _holding_period_metrics(orders, close)

    assert metrics["Completed Trade Lots"] == 3
    assert metrics["Average Holding Period (sessions)"] == pytest.approx(3.0)
    assert metrics["Median Holding Period (sessions)"] == pytest.approx(3.0)
    assert metrics["Weighted Average Holding Period (sessions)"] == pytest.approx(2.6)
    assert metrics["Completed Positions"] == 1
    assert metrics["Average Position Duration (sessions)"] == 4
    assert metrics["Open Lots"] == 0
    assert metrics["Open Positions"] == 0


def test_holding_period_unweighted_and_weighted_for_unequal_sale_quantities():
    dates = pd.bdate_range("2024-01-02", periods=6)
    close = pd.DataFrame({"A": [10.0] * len(dates)}, index=dates)
    orders = pd.DataFrame(
        [
            (0, "A", dates[0], 1.0, 10.0, 0.0, "Buy"),
            (1, "A", dates[1], 9.0, 10.0, 0.0, "Buy"),
            (2, "A", dates[4], 5.0, 10.0, 0.0, "Sell"),
            (3, "A", dates[5], 5.0, 10.0, 0.0, "Sell"),
        ],
        columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"],
    )

    metrics = _holding_period_metrics(orders, close)

    assert metrics["Completed Trade Lots"] == 3
    assert metrics["Average Holding Period (sessions)"] == pytest.approx(11 / 3)
    assert metrics["Median Holding Period (sessions)"] == pytest.approx(4.0)
    assert metrics["Weighted Average Holding Period (sessions)"] == pytest.approx(3.6)


def test_holding_period_reports_open_lots_and_excludes_them_from_completed_mean():
    dates = pd.bdate_range("2024-01-02", periods=6)
    close = pd.DataFrame({"A": [10.0] * 6, "B": [20.0] * 6}, index=dates)
    orders = pd.DataFrame(
        [
            (0, "A", dates[0], 10.0, 10.0, 0.0, "Buy"),
            (1, "A", dates[1], 4.0, 10.0, 0.0, "Sell"),
            (2, "A", dates[2], 3.0, 10.0, 0.0, "Buy"),
            (3, "B", dates[2], 4.0, 20.0, 0.0, "Buy"),
        ],
        columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"],
    )

    metrics = _holding_period_metrics(orders, close)

    assert metrics["Completed Trade Lots"] == 1
    assert metrics["Average Holding Period (sessions)"] == 1
    assert metrics["Weighted Average Holding Period (sessions)"] == 1
    assert metrics["Open Lots"] == 3
    assert metrics["Open Positions"] == 2
    assert metrics["Average Open Position Duration (sessions)"] == 4


def test_holding_period_rejects_sell_quantity_above_open_fifo_lots():
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame({"A": [10.0] * len(dates)}, index=dates)
    orders = pd.DataFrame(
        [
            (0, "A", dates[0], 2.0, 10.0, 0.0, "Buy"),
            (1, "A", dates[1], 3.0, 10.0, 0.0, "Sell"),
        ],
        columns=["Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side"],
    )

    with pytest.raises(ValueError, match="exceeds open FIFO lots"):
        _holding_period_metrics(orders, close)


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
        max_position_weight=1.0,
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
    initial = pd.Series({"EUROPE": 0.4, "USA": 0.4})
    config = MomentumConfig(
        topn=2,
        max_position_weight=1.0,
        bullish_exposure=0.5,
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
    assert weights.loc["2024-05-03", "USA"] == pytest.approx(0.5)
    held = initial.copy()
    for date in prices.index:
        orders = weights.loc[date].dropna()
        held.loc[orders.index] = orders
        assert held.sum() <= 1.0 + 1e-12
        assert held["EUROPE"] <= 0.5 + 1e-12
        assert held["USA"] <= 0.5 + 1e-12
    assert held.to_dict() == pytest.approx({"EUROPE": 0.2, "USA": 0.5})


def test_pending_orders_survive_overlapping_weekly_rebalances(monkeypatch):
    dates = pd.bdate_range("2024-06-28", "2024-07-19")
    close = pd.DataFrame(100.0, index=dates, columns=["EUROPE", "USA"])
    scores = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    signals = pd.to_datetime(["2024-06-28", "2024-07-05", "2024-07-12"])
    first_signal, second_signal, third_signal = signals
    scores.loc[signals, :] = 1.0
    volatility = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    volatility.loc[signals, :] = [4.0, 1.0]
    events = {
        (pd.Timestamp("2024-07-08"), first_signal): ["EUROPE"],
        (pd.Timestamp("2024-07-09"), first_signal): ["USA"],
        (pd.Timestamp("2024-07-10"), second_signal): ["EUROPE"],
        (pd.Timestamp("2024-07-11"), second_signal): ["USA"],
        (pd.Timestamp("2024-07-15"), third_signal): ["EUROPE"],
        (pd.Timestamp("2024-07-16"), third_signal): ["USA"],
    }
    monkeypatch.setattr(
        "pytrading.strategies.momentum._asset_execution_events",
        lambda *_args, **_kwargs: events,
    )
    initial = pd.Series({"EUROPE": 0.8, "USA": 0.2})
    config = MomentumConfig(
        topn=2,
        rebalance_frequency="weekly",
        max_position_weight=1.0,
    )

    weights = _portfolio_weights(
        close,
        close.loc[first_signal:],
        scores,
        None,
        config,
        signals=frozenset("C"),
        volatility=volatility,
        initial_weights=initial,
    )

    assert weights.loc["2024-07-08", "EUROPE"] == pytest.approx(0.2)
    assert weights.loc["2024-07-09", "USA"] == pytest.approx(0.8)
    assert pd.isna(weights.loc["2024-07-10", "EUROPE"])
    assert pd.isna(weights.loc["2024-07-11", "USA"])
    assert weights.loc["2024-07-15":"2024-07-16"].isna().all().all()
    held = initial.copy()
    for date in weights.index:
        orders = weights.loc[date].dropna()
        held.loc[orders.index] = orders
        assert held.sum() <= 1.0 + 1e-12
    assert held.to_dict() == pytest.approx({"EUROPE": 0.2, "USA": 0.8})


@pytest.mark.parametrize(
    ("second_volatility", "expected_orders"),
    [
        ([2.0, 3.0], {"EUROPE": 0.6, "USA": 0.4}),
        ([1.25, 5.0], {}),
    ],
    ids=["replace-obsolete-target", "cancel-target-already-held"],
)
def test_new_weekly_signal_replaces_or_cancels_pending_orders(
    monkeypatch, second_volatility, expected_orders
):
    dates = pd.bdate_range("2024-06-28", "2024-07-12")
    close = pd.DataFrame(100.0, index=dates, columns=["EUROPE", "USA"])
    scores = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    first_signal, second_signal = pd.Timestamp("2024-06-28"), pd.Timestamp("2024-07-05")
    scores.loc[[first_signal, second_signal], :] = 1.0
    volatility = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    volatility.loc[first_signal] = [4.0, 1.0]
    volatility.loc[second_signal] = second_volatility
    events = {
        (pd.Timestamp("2024-07-08"), first_signal): ["EUROPE"],
        (pd.Timestamp("2024-07-09"), first_signal): ["USA"],
        (pd.Timestamp("2024-07-10"), second_signal): ["EUROPE"],
        (pd.Timestamp("2024-07-11"), second_signal): ["USA"],
    }
    monkeypatch.setattr(
        "pytrading.strategies.momentum._asset_execution_events",
        lambda *_args, **_kwargs: events,
    )
    initial = pd.Series({"EUROPE": 0.8, "USA": 0.2})
    config = MomentumConfig(
        topn=2,
        rebalance_frequency="weekly",
        max_position_weight=1.0,
    )

    weights = _portfolio_weights(
        close,
        close.loc[first_signal:],
        scores,
        None,
        config,
        signals=frozenset("C"),
        volatility=volatility,
        initial_weights=initial,
    )

    assert weights.loc["2024-07-08":"2024-07-09"].isna().all().all()
    if expected_orders:
        assert weights.loc["2024-07-10", "EUROPE"] == pytest.approx(
            expected_orders["EUROPE"]
        )
        assert weights.loc["2024-07-11", "USA"] == pytest.approx(expected_orders["USA"])
    else:
        assert weights.isna().all().all()


def test_all_variants_run_identically_scheduled_vectorbt_engine(caplog):
    import logging

    close = _price_panel()
    benchmark = pd.Series(np.linspace(80, 150, len(close)), index=close.index)
    portfolio = _portfolio(close, {"USA": benchmark, "EUROPE": benchmark})
    config = MomentumConfig(topn=2, min_avg_daily_turnover=0)

    with caplog.at_level(logging.INFO, logger="pytrading.strategies.momentum"):
        report = variant_experiment(portfolio, config)

    assert list(report["Variant"]) == [variant.value for variant in StrategyVariant]
    assert report["Start Date"].nunique() == 1
    assert report["End Date"].nunique() == 1
    assert {
        "Initial Capital",
        "Final Portfolio Value",
        "Gross Final Portfolio Value",
        "CAGR",
        "Total Return",
        "Net CAGR",
        "Net Total Return",
        "Gross CAGR",
        "Gross Total Return",
        "Net End Value",
        "Gross End Value",
        "Gross-Net End Value Difference",
        "Maximum Drawdown",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Average Realized Volatility",
        "Average Position Weight",
        "Median Position Weight",
        "Largest Position Weight",
        "Average Position Volatility",
        "Weighted Average Portfolio Volatility",
        "Weight Concentration HHI",
        "Time at or Above Position Cap",
        "Downside Volatility",
        "Largest Loss Month",
        "Negative Months",
        "Positive Month Share",
        "Profit Factor",
        "Number of Trades",
        "Average Holding Period (sessions)",
        "Median Holding Period (sessions)",
        "Weighted Average Holding Period (sessions)",
        "Average Position Duration (sessions)",
        "Open Lots",
        "Open Positions",
        "Average Open Position Duration (sessions)",
        "Average Exposure",
        "Maximum Exposure",
        "Cash Allocation",
        "Time with Cash",
        "Turnover",
        "Annual Portfolio Turnover",
        "One-sided Turnover",
        "Two-sided Turnover",
        "Annualized One-sided Turnover",
        "Annualized Two-sided Turnover",
        "Transaction Costs",
        "Proportional Fees Costs",
        "Fixed Fees Costs",
        "Transaction Fees",
        "Slippage Costs",
        "Annual Transaction Costs",
        "Buy Orders",
        "Sell Orders",
        "Average EUROPE Exposure",
        "Maximum USA Exposure",
    }.issubset(report.columns)
    assert report["Buy Orders"].ge(0).all()
    assert report["Sell Orders"].ge(0).all()
    assert report["Gross Total Return"].notna().all()
    assert report["position_sizing"].tolist() == [
        "equal",
        "equal",
        "inverse_volatility",
        "equal",
        "inverse_volatility",
    ]
    assert "5 Prozesse" in caplog.text
    assert "10/10 Backtests" in caplog.text
    pd.testing.assert_frame_equal(report, variant_experiment(portfolio, config))


def test_variant_net_and_gross_reuse_prepared_backtest_inputs(monkeypatch):
    import pytrading.strategies.momentum as momentum_module

    portfolio = _portfolio(_price_panel())
    config = MomentumConfig(topn=2, min_avg_daily_turnover=0)
    monkeypatch.setattr(momentum_module, "_VARIANT_WORKER_PORTFOLIO", portfolio)
    monkeypatch.setattr(momentum_module, "_VARIANT_WORKER_CONFIG", config)
    monkeypatch.setattr(momentum_module, "_VARIANT_WORKER_FINGERPRINT", "test")
    monkeypatch.setattr(
        momentum_module,
        "_VARIANT_WORKER_UNIVERSE_SIZE",
        len(portfolio.universe.close().columns),
    )
    original_build = momentum_module._build_vectorbt_portfolio
    simulation_runs = []

    def capture_simulation(close, open_prices, weights, broker):
        result = original_build(close, open_prices, weights, broker)
        simulation_runs.append((weights, broker, result))
        return result

    monkeypatch.setattr(
        momentum_module, "_build_vectorbt_portfolio", capture_simulation
    )

    prepare = momentum_module._prepare_backtest_inputs
    calls = 0

    def count_preparations(*args, **kwargs):
        nonlocal calls
        calls += 1
        return prepare(*args, **kwargs)

    monkeypatch.setattr(momentum_module, "_prepare_backtest_inputs", count_preparations)
    result = momentum_module._run_variant_case(StrategyVariant.A)

    assert calls == 1
    assert len(simulation_runs) == 2
    net_weights, net_broker, net_result = simulation_runs[0]
    gross_weights, gross_broker, gross_result = simulation_runs[1]
    assert net_weights is gross_weights
    assert (net_broker.fees, net_broker.fixed_fees, net_broker.slippage) == (
        portfolio.broker.fees,
        portfolio.broker.fixed_fees,
        portfolio.broker.slippage,
    )
    assert (gross_broker.fees, gross_broker.fixed_fees, gross_broker.slippage) == (
        0.0,
        0.0,
        0.0,
    )
    net_orders = net_result.orders.records_readable
    gross_orders = gross_result.orders.records_readable
    assert net_orders["Fees"].sum() > 0
    assert gross_orders["Fees"].sum() == 0
    identity_columns = ["Timestamp", "Column", "Side"]
    pd.testing.assert_frame_equal(
        net_orders[identity_columns].reset_index(drop=True),
        gross_orders[identity_columns].reset_index(drop=True),
    )
    assert result["Transaction Fees"] == pytest.approx(net_orders["Fees"].sum())
    assert result["Transaction Fees"] == pytest.approx(
        result["Proportional Fees Costs"] + result["Fixed Fees Costs"]
    )
    assert result["Proportional Fees Costs"] > 0
    assert result["Fixed Fees Costs"] > 0
    assert result["Slippage Costs"] > 0
    assert result["Transaction Costs"] == pytest.approx(
        result["Transaction Fees"] + result["Slippage Costs"]
    )
    assert result["Net End Value"] == pytest.approx(net_result.value().iloc[-1])
    assert result["Gross End Value"] == pytest.approx(gross_result.value().iloc[-1])
    assert result["Gross-Net End Value Difference"] == pytest.approx(
        result["Gross End Value"] - result["Net End Value"]
    )


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
