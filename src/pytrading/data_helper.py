from itertools import starmap
from pathlib import Path

import pandas as pd
import yaml

from pytrading.bin import FX_TICKERS


def _currency_code(currency: str) -> tuple[str, float]:
    if currency == "GBp":
        return "GBP", 100.0
    return str(currency).strip().upper(), 1.0


def _convert_to_eur(
    values: pd.Series, currency: str, fx_rates: pd.DataFrame
) -> pd.Series:
    code, divisor = _currency_code(currency)
    if code == "EUR":
        return values
    if code not in FX_TICKERS:
        raise ValueError(f"Keine EUR-Umrechnung für Notierungswährung {currency!r}.")

    fx_rate = fx_rates[code].reindex(values.index).ffill()
    if fx_rate.isna().any():
        first_missing = fx_rate.index[fx_rate.isna()][0]
        raise ValueError(
            f"Kein EUR/{code}-Wechselkurs für {first_missing.date()} oder davor."
        )
    return values.div(divisor).div(fx_rate)


class DataSet:
    def __init__(self, infofile: Path, csvfile: Path):
        self.info = yaml.safe_load(infofile.open())
        self.currency = self.info.get("currency")
        if not self.currency:
            raise ValueError(f"Notierungswährung fehlt in {infofile}.")
        self.prices = pd.read_csv(csvfile)
        self.isin = infofile.stem
        if "Date" in self.prices.columns:
            self.prices["Date"] = pd.to_datetime(self.prices["Date"])
            self.prices = self.prices.set_index("Date")
        else:
            self.prices = pd.read_csv(
                csvfile,
                skiprows=2,
                parse_dates=["Date"],
                index_col="Date",
            )

    def __getattr__(self, name):
        try:
            return self.info[name]
        except KeyError:
            raise AttributeError(name) from None


class Universe:
    def __init__(self):
        data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir = data_dir
        self._fx_rates = None
        # with multiprocessing.Pool() as pool:
        self.datasets = list(
            starmap(
                DataSet,
                [
                    (file, file.with_suffix(".csv"))
                    for file in sorted(data_dir.glob("*.yaml"))
                ],
            )
        )
        self.datasets = list(
            {dataset.isin: dataset for dataset in reversed(self.datasets)}.values()
        )

    def _load_fx_rates(
        self, currencies: set[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        cache_file = self.data_dir / "fx_rates.csv"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"FX-Kursdatei fehlt: {cache_file}. Einmalig herunterladen mit "
                "`uv run python -m pytrading.bin.download_fx_rates`."
            )

        rates = pd.read_csv(cache_file, index_col="Date", parse_dates=True).sort_index()
        rates.index = pd.DatetimeIndex(rates.index.to_numpy(), name="Date")

        missing = currencies - set(rates.columns)
        if missing:
            raise ValueError(
                f"FX-Kursdatei {cache_file} enthält keine Kurse für: "
                + ", ".join(sorted(missing))
                + ". Datei mit dem FX-Downloader neu erstellen."
            )

        uncovered = []
        for currency in currencies:
            observations = rates[currency].dropna()
            if (
                observations.empty
                or observations.index[0] > start
                or observations.index[-1] < end
            ):
                uncovered.append(currency)
        if uncovered:
            raise ValueError(
                f"FX-Kursdatei {cache_file} deckt den Zeitraum "
                f"{start.date()} bis {end.date()} nicht für: "
                + ", ".join(sorted(uncovered))
                + ". Datei mit dem FX-Downloader neu erstellen."
            )
        return rates

    def _column(self, column) -> pd.DataFrame:
        if hasattr(self, f"_{column}"):
            return getattr(self, f"_{column}")

        if column in {"Close", "Turnover", "Open", "High", "Low"}:
            currencies = {
                _currency_code(ds.currency)[0]
                for ds in self.datasets
                if _currency_code(ds.currency)[0] != "EUR"
            }
            if currencies - FX_TICKERS.keys():
                raise ValueError(
                    "Keine EUR-Umrechnung für Notierungswährungen: "
                    + ", ".join(sorted(currencies - FX_TICKERS.keys()))
                )
            if currencies and self._fx_rates is None:
                start = min(ds.prices.index.min() for ds in self.datasets)
                end = max(ds.prices.index.max() for ds in self.datasets)
                self._fx_rates = self._load_fx_rates(currencies, start, end)

        values = []
        for dataset in self.datasets:
            series = dataset.prices[column]
            if column in {"Open", "High", "Low"}:
                required = {"Close", "Raw Close"}
                missing = required - set(dataset.prices.columns)
                if missing:
                    raise ValueError(
                        f"{dataset.isin} cannot align {column} to adjusted Close; "
                        f"missing columns: {', '.join(sorted(missing))}."
                    )
                raw_close = dataset.prices["Raw Close"].replace(0, pd.NA)
                adjustment = dataset.prices["Close"].div(raw_close)
                series = series.mul(adjustment)
            code, _ = _currency_code(dataset.currency)
            if column in {"Close", "Turnover", "Open", "High", "Low"} and code != "EUR":
                series = _convert_to_eur(series, dataset.currency, self._fx_rates)
            values.append(series)

        data = pd.concat(
            values,
            axis=1,
            keys=[ds.isin for ds in self.datasets],
        )
        setattr(self, f"_{column}", data)
        return data

    def close(self) -> pd.DataFrame:
        return self._column("Close")

    def turnover(self) -> pd.DataFrame:
        return self._column("Turnover")

    def open(self) -> pd.DataFrame:
        return self._column("Open")

    def volume(self) -> pd.DataFrame:
        return self._column("Volume")

    def high(self) -> pd.DataFrame:
        return self._column("High")

    def low(self) -> pd.DataFrame:
        return self._column("Low")

    def market_metadata(self) -> dict[str, dict[str, str | None]]:
        return {
            dataset.isin: {
                "country": dataset.info.get("country"),
                "exchange": dataset.info.get("exchange"),
            }
            for dataset in self.datasets
        }
