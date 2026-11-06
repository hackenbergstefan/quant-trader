import pandas as pd
import pytest

from quant_trader.data.market_data import MarketDataValidationError, normalize_prices
from quant_trader.data.market_data import missing_sessions
from quant_trader.data.storage import MarketDataStore
from quant_trader.domain.models import Instrument


@pytest.fixture
def instrument() -> Instrument:
    return Instrument("TEST", "TEST.L", "LSE", "GBP", "XLON", 1.2)


def test_normalize_accepts_timestamp_column_and_preserves_adjusted_close(
    instrument: Instrument,
) -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02T09:00:00+00:00"],
            "Open": [10],
            "High": [12],
            "Low": [9],
            "Close": [11],
            "Adj Close": [10.5],
            "Volume": [100],
        }
    )

    normalized = normalize_prices(raw, instrument, "test", pd.Timestamp("2024-01-03", tz="UTC"))

    assert normalized.loc[0, "timestamp"] == pd.Timestamp("2024-01-02")
    assert normalized.loc[0, "adjusted_close"] == 10.5
    assert normalized.loc[0, "fx_to_eur"] == 1.2


def test_normalize_rejects_duplicate_sessions(instrument: Instrument) -> None:
    raw = pd.DataFrame(
        {"Open": [10, 10], "High": [12, 12], "Low": [9, 9], "Close": [11, 11], "Volume": [1, 1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-02"]),
    )

    with pytest.raises(MarketDataValidationError, match="Duplicate sessions"):
        normalize_prices(raw, instrument, "test")


def test_normalize_uses_point_in_time_fx_rates(instrument: Instrument) -> None:
    raw = pd.DataFrame(
        {
            "Open": [10, 11],
            "High": [12, 13],
            "Low": [9, 10],
            "Close": [11, 12],
            "Volume": [10, 10],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    rates = pd.DataFrame(
        {"fx_to_eur": [1.1, 1.15, 1.2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    normalized = normalize_prices(raw, instrument, "test", fx_rates=rates)

    assert normalized["fx_to_eur"].tolist() == [1.1, 1.15]


def test_split_adjusted_provider_prices_are_restored_to_historical_units(
    instrument: Instrument,
) -> None:
    raw = pd.DataFrame(
        {
            "Open": [49, 50],
            "High": [51, 52],
            "Low": [48, 49],
            "Close": [50, 51],
            "Adj Close": [45, 46],
            "Volume": [200, 100],
            "Stock Splits": [0, 2],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    normalized = normalize_prices(raw, instrument, "test", price_adjustment="split_adjusted")

    assert normalized["close"].tolist() == [100, 51]
    assert normalized["volume"].tolist() == [100, 100]
    assert normalized["adjusted_close"].tolist() == [45, 46]


def test_missing_sessions_uses_supplied_calendar_sessions(instrument: Instrument) -> None:
    raw = pd.DataFrame(
        {
            "Open": [10, 11],
            "High": [12, 13],
            "Low": [9, 10],
            "Close": [11, 12],
            "Volume": [10, 10],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    normalized = normalize_prices(raw, instrument, "test")

    missing = missing_sessions(
        normalized,
        instrument.instrument_id,
        pd.date_range("2024-01-02", "2024-01-04", freq="B"),
    )

    assert missing == [pd.Timestamp("2024-01-03").date()]


def test_duckdb_keeps_raw_payload_separate_from_normalized_prices(
    instrument: Instrument, tmp_path
) -> None:
    raw = pd.DataFrame(
        {
            "Open": [10],
            "High": [12],
            "Low": [9],
            "Close": [11],
            "Adj Close": [10.5],
            "Volume": [100],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    normalized = normalize_prices(raw, instrument, "test")

    with MarketDataStore(tmp_path / "market.duckdb") as store:
        store.save(raw, normalized)
        stored = store.load_prices()
        payload = store.connection.execute("SELECT payload FROM raw_prices").fetchone()[0]

    assert stored.loc[0, "adjusted_close"] == 10.5
    assert '"adj_close": 10.5' in payload
