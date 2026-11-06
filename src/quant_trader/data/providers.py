from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from quant_trader.domain.models import Instrument


class HistoricalPriceProvider(Protocol):
    name: str

    def fetch_history(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame: ...


class MarketDataProvider(HistoricalPriceProvider, Protocol):
    """Provider contract; strategy code depends on normalized data, not a vendor."""


class YahooFinanceProvider:
    name = "Yahoo Finance"
    price_adjustment = "split_adjusted"

    def fetch_history(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install quant-trader with its yfinance dependency") from exc

        try:
            frame = yf.Ticker(instrument.symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Yahoo Finance request failed for {instrument.symbol}: {exc}"
            ) from exc
        if frame.empty:
            raise ValueError(f"Yahoo Finance returned no history for {instrument.symbol}")
        return frame

    def fetch_fx_rates(self, currency: str, start: date, end: date) -> pd.DataFrame:
        if currency.upper() == "EUR":
            return pd.DataFrame({"fx_to_eur": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp(start)]))
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install quant-trader with its yfinance dependency") from exc
        ticker = f"{currency.upper()}EUR=X"
        frame = yf.Ticker(ticker).history(
            start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), auto_adjust=False
        )
        if frame.empty or "Close" not in frame:
            raise ValueError(f"Yahoo Finance returned no EUR FX rates for {currency}")
        return frame[["Close"]].rename(columns={"Close": "fx_to_eur"})


def extract_corporate_actions(raw: pd.DataFrame, instrument: Instrument) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["instrument_id", "ex_date", "action_type", "value", "currency"]
        )
    actions: list[dict] = []
    for timestamp, row in raw.iterrows():
        day = pd.Timestamp(timestamp)
        if day.tzinfo is not None:
            day = day.tz_localize(None)
        dividend = float(row.get("Dividends", 0) or 0)
        split = float(row.get("Stock Splits", 0) or 0)
        if dividend:
            actions.append(
                {
                    "instrument_id": instrument.instrument_id,
                    "ex_date": day.date(),
                    "action_type": "dividend",
                    "value": dividend,
                    "currency": instrument.currency,
                }
            )
        if split:
            actions.append(
                {
                    "instrument_id": instrument.instrument_id,
                    "ex_date": day.date(),
                    "action_type": "split",
                    "value": split,
                    "currency": instrument.currency,
                }
            )
    return pd.DataFrame(actions)
