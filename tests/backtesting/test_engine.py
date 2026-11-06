from datetime import date

import pandas as pd
import pytest

from quant_trader.backtesting.engine import BacktestEngine
from quant_trader.config import ExecutionConfig, RiskConfig
from quant_trader.domain.models import ActionType, CorporateAction


def bars_for(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=len(prices), freq="B")
    return pd.DataFrame(
        [
            {
                "instrument_id": "A",
                "symbol": "A.DE",
                "timestamp": stamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": close,
                "volume": 100_000,
                "fx_to_eur": 1.0,
            }
            for stamp, (open_price, high, low, close) in zip(dates, prices, strict=True)
        ]
    )


def signals_for(bars: pd.DataFrame, selected: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": "A",
            "timestamp": bars["timestamp"],
            "target_weight": [0.2 if value else 0.0 for value in selected],
            "selected": selected,
            "atr": 2.0,
            "momentum_score": 0.1,
        }
    )


def make_engine() -> BacktestEngine:
    risk = RiskConfig(
        starting_cash_eur=5_000,
        max_position_value_eur=1_000,
        max_portfolio_exposure=1,
        risk_budget_per_position=0.01,
    )
    execution = ExecutionConfig(
        commission_bps=1, exchange_fee_bps=0.5, spread_bps=5, slippage_bps=2
    )
    return BacktestEngine(risk, execution, stop_atr_multiple=2)


def test_close_signal_executes_at_next_open_and_keeps_cash_nonnegative() -> None:
    bars = bars_for([(100, 101, 99, 100), (102, 104, 100, 103), (105, 106, 103, 104)])
    signals = signals_for(bars, [True, False, False])

    result = make_engine().run(bars, signals)

    assert result.transactions.iloc[0]["timestamp"] == date(2024, 1, 3)
    assert result.transactions.iloc[0]["side"] == "buy"
    assert result.transactions.iloc[0]["price"] == 102
    assert result.equity_curve.iloc[0]["cash"] == 5_000
    assert (result.equity_curve["cash"] >= 0).all()
    assert (result.transactions["cash_after_eur"] >= -1e-8).all()


def test_gap_through_stop_fills_at_open_and_stays_out_until_deselected() -> None:
    bars = bars_for([(100, 101, 99, 100), (100, 102, 99, 101), (94, 98, 93, 95), (96, 98, 95, 97)])
    signals = signals_for(bars, [True, True, True, False])

    result = make_engine().run(bars, signals)

    stop = result.transactions.loc[result.transactions["reason"] == "stop_gap"]
    assert len(stop) == 1
    assert stop.iloc[0]["timestamp"] == date(2024, 1, 4)
    assert stop.iloc[0]["price"] == 94
    assert (result.equity_curve["cash"] >= 0).all()


def test_split_and_dividend_are_applied_to_open_positions() -> None:
    bars = bars_for(
        [(100, 101, 99, 100), (100, 101, 99, 100), (51, 52, 50, 51), (50.5, 51, 50, 50.5)]
    )
    signals = signals_for(bars, [True, True, True, True])
    signals["atr"] = 1.0
    actions = [
        CorporateAction("A", date(2024, 1, 4), ActionType.SPLIT, 2.0, "EUR"),
        CorporateAction("A", date(2024, 1, 5), ActionType.DIVIDEND, 0.5, "EUR"),
    ]

    result = make_engine().run(bars, signals, actions)

    assert len(result.transactions.loc[result.transactions["side"] == "buy"]) == 1
    assert result.transactions.iloc[-1]["reason"] == "terminal_liquidation"
    assert result.trades.iloc[0]["pnl_eur"] == pytest.approx(18.2915)
    assert (result.equity_curve["cash"] >= 0).all()


def test_backtest_is_deterministic() -> None:
    bars = bars_for([(100, 101, 99, 100), (102, 104, 100, 103), (105, 106, 103, 104)])
    signals = signals_for(bars, [True, False, False])
    engine = make_engine()

    first = engine.run(bars, signals)
    second = engine.run(bars, signals)

    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    pd.testing.assert_frame_equal(first.transactions, second.transactions)


def test_fill_time_position_cap_prioritizes_higher_ranked_candidate() -> None:
    single = bars_for([(100, 101, 99, 100), (100, 102, 99, 101), (101, 102, 100, 101)])
    bars = pd.concat(
        [
            single.assign(instrument_id=instrument_id, symbol=f"{instrument_id}.DE")
            for instrument_id in ("A", "B")
        ],
        ignore_index=True,
    )
    signals = pd.DataFrame(
        {
            "instrument_id": bars["instrument_id"],
            "timestamp": bars["timestamp"],
            "target_weight": 0.5,
            "selected": True,
            "atr": 2.0,
            "momentum_score": [0.1 if item == "A" else 0.9 for item in bars["instrument_id"]],
        }
    )
    engine = BacktestEngine(
        RiskConfig(starting_cash_eur=5_000, max_position_value_eur=1_000),
        ExecutionConfig(),
        max_positions=1,
    )

    result = engine.run(bars, signals)

    buys = result.transactions.loc[result.transactions["side"] == "buy"]
    assert buys["instrument_id"].tolist() == ["B"]
    assert buys.iloc[0]["notional_eur"] <= 1_000


def test_split_rounds_down_and_pays_fractional_share_in_cash() -> None:
    bars = bars_for([(110, 111, 109, 110), (110, 111, 109, 110), (73.5, 74, 73, 73.5)])
    signals = signals_for(bars, [True, True, True])
    signals["atr"] = 1.0
    action = CorporateAction("A", date(2024, 1, 4), ActionType.SPLIT, 1.5, "EUR")

    result = make_engine().run(bars, signals, [action])

    assert result.transactions.iloc[0]["quantity"] == 9
    cash_in_lieu = result.transactions.loc[result.transactions["side"] == "cash_in_lieu"]
    assert len(cash_in_lieu) == 1
    assert cash_in_lieu.iloc[0]["quantity"] == 0.5
    assert cash_in_lieu.iloc[0]["price"] == 73.5
    assert result.transactions.iloc[-1]["quantity"] == 13
