from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

import pandas as pd

from quant_trader.domain.models import Instrument

NORMALIZED_COLUMNS = [
    "instrument_id",
    "symbol",
    "exchange",
    "currency",
    "calendar",
    "fx_to_eur",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "data_source",
    "retrieved_at",
]


class MarketDataValidationError(ValueError):
    """Raised when input bars cannot safely be used for research."""


def normalize_prices(
    raw: pd.DataFrame,
    instrument: Instrument,
    source: str,
    retrieved_at: datetime | None = None,
    fx_rates: pd.DataFrame | None = None,
    price_adjustment: Literal["unadjusted", "split_adjusted"] = "unadjusted",
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame = raw.copy()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame.pop("timestamp"), errors="coerce")
    else:
        timestamps = pd.to_datetime(frame.index, errors="coerce")
    if isinstance(timestamps, pd.DatetimeIndex) and timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)
    elif isinstance(timestamps, pd.Series) and timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)
    if "adj_close" in frame.columns and "adjusted_close" not in frame.columns:
        frame = frame.rename(columns={"adj_close": "adjusted_close"})
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise MarketDataValidationError(f"Missing required OHLCV columns: {sorted(missing)}")
    if "adjusted_close" not in frame.columns:
        frame["adjusted_close"] = float("nan")

    frame = frame.reset_index(drop=True)
    parsed_timestamps = pd.to_datetime(timestamps, errors="coerce")
    frame["timestamp"] = (
        parsed_timestamps.dt.normalize()
        if isinstance(parsed_timestamps, pd.Series)
        else parsed_timestamps.normalize()
    )
    for column in ("open", "high", "low", "close", "adjusted_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if price_adjustment == "split_adjusted":
        split_values = pd.to_numeric(
            frame.get("stock_splits", pd.Series(0.0, index=frame.index)), errors="coerce"
        ).fillna(0.0)
        split_factor = pd.Series(1.0, index=frame.index)
        for split_date, ratio in zip(frame["timestamp"], split_values, strict=True):
            if ratio > 0 and ratio != 1:
                split_factor.loc[frame["timestamp"] < split_date] *= ratio
        frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].mul(
            split_factor, axis=0
        )
        frame["volume"] = frame["volume"].div(split_factor)
    elif price_adjustment != "unadjusted":
        raise ValueError(f"Unsupported price adjustment convention: {price_adjustment}")
    frame["instrument_id"] = instrument.instrument_id
    frame["symbol"] = instrument.symbol
    frame["exchange"] = instrument.exchange
    frame["currency"] = instrument.currency
    frame["calendar"] = instrument.calendar
    frame["fx_to_eur"] = instrument.fx_to_eur
    if fx_rates is not None:
        rates = fx_rates.copy()
        if "timestamp" in rates.columns:
            rate_timestamps = pd.to_datetime(rates.pop("timestamp"), errors="coerce")
        else:
            rate_timestamps = pd.to_datetime(rates.index, errors="coerce")
        if isinstance(rate_timestamps, pd.Series):
            if rate_timestamps.dt.tz is not None:
                rate_timestamps = rate_timestamps.dt.tz_localize(None)
            rate_timestamps = rate_timestamps.dt.normalize()
        else:
            if rate_timestamps.tz is not None:
                rate_timestamps = rate_timestamps.tz_localize(None)
            rate_timestamps = rate_timestamps.normalize()
        rate_column = "fx_to_eur" if "fx_to_eur" in rates else "close"
        rates = pd.DataFrame(
            {
                "timestamp": rate_timestamps,
                "fx_to_eur": pd.to_numeric(rates[rate_column], errors="coerce"),
            }
        )
        rates = rates.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        frame = pd.merge_asof(
            frame.sort_values("timestamp"),
            rates,
            on="timestamp",
            direction="backward",
            allow_exact_matches=False,
            suffixes=("", "_rate"),
        )
        frame["fx_to_eur"] = frame["fx_to_eur_rate"]
        frame = frame.drop(columns="fx_to_eur_rate")
    frame["data_source"] = source
    frame["retrieved_at"] = (retrieved_at or datetime.now(timezone.utc)).isoformat()
    frame = frame[NORMALIZED_COLUMNS].sort_values("timestamp", kind="stable").reset_index(drop=True)

    if frame["timestamp"].isna().any():
        raise MarketDataValidationError(f"Invalid timestamp for {instrument.instrument_id}")
    if frame["fx_to_eur"].isna().any() or (frame["fx_to_eur"] <= 0).any():
        raise MarketDataValidationError(
            f"Missing or invalid EUR FX rate for {instrument.instrument_id}"
        )
    if frame["timestamp"].duplicated().any():
        duplicates = frame.loc[
            frame["timestamp"].duplicated(keep=False), "timestamp"
        ].dt.date.tolist()
        raise MarketDataValidationError(
            f"Duplicate sessions for {instrument.instrument_id}: {duplicates}"
        )
    required_prices = frame[["open", "high", "low", "close", "volume"]]
    if required_prices.isna().any().any() or (required_prices < 0).any().any():
        raise MarketDataValidationError(
            f"Null or negative OHLCV values for {instrument.instrument_id}"
        )
    invalid = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame["close"] <= 0)
        | (frame["open"] <= 0)
        | ((frame["adjusted_close"].notna()) & (frame["adjusted_close"] <= 0))
    )
    if invalid.any():
        raise MarketDataValidationError(f"Invalid OHLC relationship for {instrument.instrument_id}")
    return frame


def missing_sessions(
    prices: pd.DataFrame, instrument_id: str, expected_sessions: pd.DatetimeIndex
) -> list[date]:
    observed = pd.DatetimeIndex(
        pd.to_datetime(prices.loc[prices["instrument_id"] == instrument_id, "timestamp"])
    ).normalize()
    expected = (
        expected_sessions.tz_localize(None).normalize()
        if expected_sessions.tz is not None
        else expected_sessions.normalize()
    )
    return [stamp.date() for stamp in expected.difference(observed)]
