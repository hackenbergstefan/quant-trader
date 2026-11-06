from __future__ import annotations

import pandas as pd

from quant_trader.backtesting.engine import BacktestResult, calculate_metrics


def buy_and_hold_benchmark(
    reference: pd.DataFrame,
    starting_cash_eur: float,
    cost_rate: float,
    name: str = "EURO STOXX 50 buy and hold",
) -> BacktestResult:
    frame = reference.sort_values("timestamp").copy()
    price_column = (
        "adjusted_close"
        if "adjusted_close" in frame and frame["adjusted_close"].notna().all()
        else "close"
    )
    frame = frame.dropna(subset=[price_column]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Benchmark data is unavailable")
    prices = frame[price_column].astype(float)
    equity_values = starting_cash_eur * (1 - cost_rate) * prices / prices.iloc[0]
    terminal_notional = float(equity_values.iloc[-1])
    equity_values.iloc[-1] = terminal_notional * (1 - cost_rate)
    cumulative_costs = pd.Series(0.0, index=frame.index)
    cumulative_costs.iloc[0] = starting_cash_eur * cost_rate
    cumulative_costs.iloc[-1] += terminal_notional * cost_rate
    curve = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"]),
            "equity": equity_values,
            "cash": 0.0,
            "exposure_eur": equity_values,
            "exposure_ratio": 1.0,
            "transaction_costs_eur": cumulative_costs,
        }
    )
    curve["daily_return"] = curve["equity"].pct_change().fillna(0.0)
    transactions = pd.DataFrame(
        [
            {
                "timestamp": curve.iloc[0]["timestamp"].date(),
                "instrument_id": str(frame.iloc[0].get("instrument_id", name)),
                "side": "buy",
                "quantity": 1,
                "price": float(prices.iloc[0]),
                "notional_eur": starting_cash_eur,
                "costs_eur": starting_cash_eur * cost_rate,
                "reason": "benchmark_entry",
                "cash_after_eur": 0.0,
            },
            {
                "timestamp": curve.iloc[-1]["timestamp"].date(),
                "instrument_id": str(frame.iloc[0].get("instrument_id", name)),
                "side": "sell",
                "quantity": 1,
                "price": float(prices.iloc[-1]),
                "notional_eur": terminal_notional,
                "costs_eur": terminal_notional * cost_rate,
                "reason": "benchmark_valuation",
                "cash_after_eur": 0.0,
            },
        ]
    )
    trades = pd.DataFrame()
    metadata = {
        "strategy": name,
        "start_date": curve["timestamp"].min().date().isoformat(),
        "end_date": curve["timestamp"].max().date().isoformat(),
        "benchmark_method": "adjusted close total-return proxy; entry and terminal liquidation costs applied",
    }
    metrics = calculate_metrics(curve, transactions, trades, initial_capital=starting_cash_eur)
    metrics["total_return_gross_estimate"] = float(prices.iloc[-1] / prices.iloc[0] - 1)
    return BacktestResult(
        curve,
        transactions,
        trades,
        metrics,
        metadata,
    )


def cash_benchmark(starting_cash_eur: float, dates: pd.Series) -> BacktestResult:
    timestamps = pd.to_datetime(dates).drop_duplicates().sort_values().reset_index(drop=True)
    curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "equity": starting_cash_eur,
            "cash": starting_cash_eur,
            "exposure_eur": 0.0,
            "exposure_ratio": 0.0,
            "transaction_costs_eur": 0.0,
            "daily_return": 0.0,
        }
    )
    metadata = {
        "strategy": "cash",
        "start_date": timestamps.min().date().isoformat(),
        "end_date": timestamps.max().date().isoformat(),
    }
    return BacktestResult(
        curve,
        pd.DataFrame(),
        pd.DataFrame(),
        calculate_metrics(curve, pd.DataFrame(), pd.DataFrame(), initial_capital=starting_cash_eur),
        metadata,
    )
