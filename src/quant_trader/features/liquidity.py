from __future__ import annotations

import pandas as pd


def calculate_average_daily_traded_value(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    frame = prices.sort_values(["instrument_id", "timestamp"]).copy()
    fx = frame["fx_to_eur"] if "fx_to_eur" in frame else 1.0
    frame["traded_value_eur"] = frame["close"] * frame["volume"] * fx
    frame["average_daily_traded_value_eur"] = frame.groupby("instrument_id", sort=False)[
        "traded_value_eur"
    ].transform(lambda values: values.rolling(window, min_periods=window).mean())
    return frame[["instrument_id", "timestamp", "average_daily_traded_value_eur"]]


def build_features(prices: pd.DataFrame, config: object) -> pd.DataFrame:
    from quant_trader.features.momentum import calculate_momentum
    from quant_trader.features.trend import calculate_trend, feature_price_view
    from quant_trader.features.volatility import calculate_realized_volatility

    view = feature_price_view(prices)
    features = view[["instrument_id", "timestamp", "feature_close"]].copy()
    for calculated in (
        calculate_momentum(view, config.momentum_periods, config.momentum_weights),
        calculate_trend(view, (config.sma_fast, config.sma_slow), config.atr_period),
        calculate_realized_volatility(view, config.volatility_periods),
        calculate_average_daily_traded_value(prices, config.liquidity_window),
    ):
        features = features.merge(
            calculated, on=["instrument_id", "timestamp"], how="left", validate="one_to_one"
        )
    return features.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True)
