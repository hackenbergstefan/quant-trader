from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_realized_volatility(
    prices: pd.DataFrame, periods: tuple[int, int] = (20, 60)
) -> pd.DataFrame:
    frame = prices.sort_values(["instrument_id", "timestamp"]).copy()
    price_column = "feature_close" if "feature_close" in frame else "close"
    frame["daily_return"] = frame.groupby("instrument_id", sort=False)[price_column].pct_change(
        fill_method=None
    )
    for period in periods:
        frame[f"volatility_{period}"] = frame.groupby("instrument_id", sort=False)[
            "daily_return"
        ].transform(
            lambda values: values.rolling(period, min_periods=period).std(ddof=1) * np.sqrt(252)
        )
    return frame[["instrument_id", "timestamp", *[f"volatility_{period}" for period in periods]]]
