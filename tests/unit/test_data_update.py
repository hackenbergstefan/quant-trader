from datetime import date

import pandas as pd

from quant_trader.data.storage import MarketDataStore
from quant_trader.data.update import update_history
from quant_trader.domain.models import Instrument


class OneInstrumentRepository:
    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument

    def list_instruments(self, active_only: bool = True) -> list[Instrument]:
        return [self.instrument]


class InvalidPriceProvider:
    name = "test source"

    def fetch_history(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame(
            {"Open": [10], "High": [9], "Low": [8], "Close": [10], "Volume": [100]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )


def test_invalid_bars_are_retained_raw_but_not_normalized(tmp_path) -> None:
    instrument = Instrument("TEST", "TEST.DE", "XETR", "EUR", "XETR")
    database = tmp_path / "market.duckdb"
    with MarketDataStore(database) as store:
        count, failures = update_history(
            OneInstrumentRepository(instrument),
            InvalidPriceProvider(),
            store,
            date(2024, 1, 1),
            date(2024, 1, 3),
        )
        raw_count = store.connection.execute("SELECT COUNT(*) FROM raw_prices").fetchone()[0]
        normalized_count = store.connection.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

    assert count == 0
    assert failures == ["TEST"]
    assert raw_count == 1
    assert normalized_count == 0
