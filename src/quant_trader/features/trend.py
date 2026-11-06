from __future__ import annotations

import numpy as np
import pandas as pd


def feature_price_view(prices: pd.DataFrame) -> pd.DataFrame:
    """Use adjusted OHLC consistently for indicators when a complete adjusted series exists."""
    pieces: list[pd.DataFrame] = []
    for _, group in prices.sort_values("timestamp").groupby("instrument_id", sort=False):
        frame = group.copy()
        adjusted = frame["adjusted_close"]
        use_adjusted = adjusted.notna().all() and (adjusted > 0).all()
        factor = adjusted.div(frame["close"]) if use_adjusted else 1.0
        frame["feature_close"] = adjusted if use_adjusted else frame["close"]
        for column in ("open", "high", "low"):
            frame[f"feature_{column}"] = frame[column] * factor
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else prices.copy()


def calculate_trend(
    prices: pd.DataFrame, sma_windows: tuple[int, int] = (50, 200), atr_period: int = 14
) -> pd.DataFrame:
    frame = prices.sort_values(["instrument_id", "timestamp"]).copy()
    if "feature_close" not in frame:
        frame = feature_price_view(frame)
    groups = frame.groupby("instrument_id", sort=False)
    for window in sma_windows:
        frame[f"sma_{window}"] = groups["feature_close"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    previous_close = groups["feature_close"].shift(1)
    true_range = pd.concat(
        [
            frame["feature_high"] - frame["feature_low"],
            (frame["feature_high"] - previous_close).abs(),
            (frame["feature_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame[f"atr_{atr_period}"] = true_range.groupby(frame["instrument_id"], sort=False).transform(
        lambda values: values.rolling(atr_period, min_periods=atr_period).mean()
    )
    # Keep stop distances in executable raw-price units after retrospective adjustments.
    frame[f"atr_{atr_period}"] *= frame["close"] / frame["feature_close"]
    return frame[
        ["instrument_id", "timestamp", *[f"sma_{n}" for n in sma_windows], f"atr_{atr_period}"]
    ]
