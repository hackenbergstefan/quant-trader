import pandas as pd

from quant_trader.config import StrategyConfig
from quant_trader.portfolio.allocation import apply_allocation_method
from quant_trader.portfolio.risk_manager import RiskManager
from quant_trader.config import RiskConfig
from quant_trader.strategies.adaptive_momentum import AdaptiveMomentumStrategy


def simple_features() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2024-01-01", periods=4, freq="B"):
        for instrument_id, score in (("A", 0.3), ("B", 0.2)):
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "timestamp": day,
                    "momentum_score": score,
                    "feature_close": 100.0,
                    "sma_1": 99.0,
                    "sma_3": 95.0,
                    "atr_2": 2.0,
                    "volatility_20": 0.2 if instrument_id == "A" else 0.4,
                    "average_daily_traded_value_eur": 2_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_cross_sectional_ranking_selects_highest_eligible_score() -> None:
    config = StrategyConfig(
        momentum_periods=(1, 2, 3),
        sma_fast=1,
        sma_slow=3,
        atr_period=2,
        min_history=3,
        max_positions=1,
    )
    features = simple_features()

    signals = AdaptiveMomentumStrategy(
        config, trend_filter=True, regime_filter=False, liquidity_filter=True
    ).generate_signals(features)
    day_three = signals.loc[signals["timestamp"] == pd.Timestamp("2024-01-03")]

    assert day_three.loc[day_three["selected"], "instrument_id"].tolist() == ["A"]
    assert day_three["target_weight"].sum() == 1.0
    assert not signals.loc[signals["timestamp"] == pd.Timestamp("2024-01-02"), "selected"].any()


def test_future_feature_changes_do_not_change_earlier_signals() -> None:
    config = StrategyConfig(
        momentum_periods=(1, 2, 3),
        sma_fast=1,
        sma_slow=3,
        atr_period=2,
        min_history=3,
        max_positions=1,
    )
    strategy = AdaptiveMomentumStrategy(
        config, trend_filter=False, regime_filter=False, liquidity_filter=False
    )
    original = simple_features()
    altered = original.copy()
    altered.loc[altered["timestamp"] == pd.Timestamp("2024-01-04"), "momentum_score"] = 100.0

    first = strategy.generate_signals(original)
    second = strategy.generate_signals(altered)
    pd.testing.assert_frame_equal(
        first.iloc[:6].reset_index(drop=True), second.iloc[:6].reset_index(drop=True)
    )


def test_allocation_modes_are_normalized_and_risk_sizing_rounds_down() -> None:
    signals = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")] * 3,
            "selected": [True, True, True],
            "target_weight": [1 / 3] * 3,
            "volatility_20": [0.1, 0.2, 0.4],
            "feature_close": [100.0, 100.0, 100.0],
            "atr": [1.0, 2.0, 1.0],
        }
    )

    volatility = apply_allocation_method(signals, "volatility_adjusted")
    risk_budget = apply_allocation_method(signals, "risk_budget")
    manager = RiskManager(RiskConfig(), 2.0, 0.001)
    shares = manager.shares_for_entry(
        equity_eur=5_000,
        cash_eur=5_000,
        entry_price=100,
        fx_to_eur=1,
        atr=5,
        target_weight=0.2,
        invested_eur=0,
    )

    assert abs(volatility["target_weight"].sum() - 1.0) < 1e-12
    assert volatility.loc[0, "target_weight"] > volatility.loc[1, "target_weight"]
    assert abs(risk_budget["target_weight"].sum() - 1.0) < 1e-12
    assert risk_budget.loc[0, "target_weight"] > risk_budget.loc[1, "target_weight"]
    assert shares == 5
