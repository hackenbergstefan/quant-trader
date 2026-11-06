from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd

from quant_trader.domain.models import Instrument


def synthetic_market_data(
    start: date = date(2010, 1, 1),
    end: date = date(2024, 12, 31),
    instrument_count: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce a deterministic test-only panel; it is not historical market data."""
    sessions = pd.bdate_range(start, end)
    bars: list[pd.DataFrame] = []
    returns: list[np.ndarray] = []
    steps = np.arange(len(sessions), dtype=float)
    for index in range(instrument_count):
        instrument_id = f"SYN{index + 1:02d}"
        seed = int.from_bytes(hashlib.sha256(instrument_id.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        drift = (index - (instrument_count - 1) / 2) * 0.000025 + 0.00015
        cycle = 0.0015 * np.sin(steps / (75 + 11 * index))
        daily_returns = drift + cycle + rng.normal(0.0, 0.009 + index * 0.0002, len(sessions))
        close = (60 + index * 9) * np.cumprod(1 + daily_returns)
        open_price = close * (1 + rng.normal(0.0, 0.002, len(sessions)))
        high = np.maximum(open_price, close) * 1.006
        low = np.minimum(open_price, close) * 0.994
        volume = np.full(len(sessions), 60_000 + index * 2_500, dtype=float)
        bars.append(
            pd.DataFrame(
                {
                    "instrument_id": instrument_id,
                    "symbol": f"{instrument_id}.EU",
                    "exchange": "SYNTHETIC",
                    "currency": "EUR",
                    "calendar": "WEEKDAY",
                    "fx_to_eur": 1.0,
                    "timestamp": sessions,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adjusted_close": close,
                    "volume": volume,
                    "data_source": "deterministic synthetic generator",
                    "retrieved_at": "fixed synthetic input",
                }
            )
        )
        returns.append(daily_returns)

    benchmark_returns = np.mean(np.vstack(returns), axis=0) * 0.65 + 0.00005
    benchmark_close = 3_000 * np.cumprod(1 + benchmark_returns)
    benchmark = pd.DataFrame(
        {
            "instrument_id": "EUROSTOXX50",
            "symbol": "^STOXX50E",
            "exchange": "SYNTHETIC",
            "currency": "EUR",
            "calendar": "WEEKDAY",
            "fx_to_eur": 1.0,
            "timestamp": sessions,
            "open": benchmark_close * 0.999,
            "high": benchmark_close * 1.004,
            "low": benchmark_close * 0.996,
            "close": benchmark_close,
            "adjusted_close": benchmark_close,
            "volume": 1_000_000,
            "data_source": "deterministic synthetic generator",
            "retrieved_at": "fixed synthetic input",
        }
    )
    return pd.concat(bars, ignore_index=True), benchmark


class SyntheticPriceProvider:
    name = "deterministic synthetic generator"

    def __init__(self, instrument_count: int = 8) -> None:
        self.instrument_count = instrument_count

    def fetch_history(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        equities, benchmark = synthetic_market_data(start, end, self.instrument_count)
        panel = benchmark if instrument.instrument_id == "EUROSTOXX50" else equities
        selected = panel.loc[panel["instrument_id"] == instrument.instrument_id]
        if selected.empty:
            raise ValueError(f"Synthetic instrument not found: {instrument.instrument_id}")
        return selected.set_index("timestamp")[
            ["open", "high", "low", "close", "adjusted_close", "volume"]
        ]
