from datetime import date
from types import SimpleNamespace

import pandas as pd

from quant_trader.data.providers import YahooFinanceProvider
from quant_trader.domain.models import Instrument


def test_yahoo_provider_translates_inclusive_end_to_exclusive_api_end(monkeypatch) -> None:
    calls = []

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs):
            calls.append((self.symbol, kwargs))
            return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setitem(__import__("sys").modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))
    provider = YahooFinanceProvider()
    instrument = Instrument("SAP.DE", "SAP.DE", "XETR", "EUR", "XETR")

    provider.fetch_history(instrument, date(2024, 1, 1), date(2024, 1, 3))
    provider.fetch_fx_rates("SEK", date(2024, 1, 1), date(2024, 1, 3))

    assert calls[0][1]["start"] == "2024-01-01"
    assert calls[0][1]["end"] == "2024-01-04"
    assert calls[1][1]["end"] == "2024-01-04"
