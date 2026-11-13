import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from pytrading.bin import FX_TICKERS


def download_fx_rates(
    currencies: set[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    unsupported = currencies - FX_TICKERS.keys()
    if unsupported:
        raise ValueError(
            "Nicht unterstützte Währungen: " + ", ".join(sorted(unsupported))
        )

    ticker_to_currency = {FX_TICKERS[currency]: currency for currency in currencies}
    raw = yf.download(
        tickers=list(ticker_to_currency),
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Historische EUR-Wechselkurse konnten nicht geladen werden.")

    rates = {}
    for ticker, currency in ticker_to_currency.items():
        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close", ticker) in raw.columns:
                rates[currency] = raw[("Close", ticker)]
            elif (ticker, "Close") in raw.columns:
                rates[currency] = raw[(ticker, "Close")]
        elif len(ticker_to_currency) == 1 and "Close" in raw.columns:
            rates[currency] = raw["Close"]

    missing = currencies - rates.keys()
    if missing:
        raise RuntimeError("EUR-Wechselkurse fehlen für: " + ", ".join(sorted(missing)))

    result = pd.DataFrame(rates)
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    return result.sort_index()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Download historical EUR FX rates.")
    parser.add_argument(
        "--currencies",
        nargs="+",
        choices=sorted(FX_TICKERS),
        default=sorted(FX_TICKERS),
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--output", type=Path, default=project_root / "data" / "fx_rates.csv"
    )
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        parser.error("--end muss gleich oder nach --start liegen")

    rates = download_fx_rates(set(args.currencies), start, end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(args.output, index_label="Date")
    print(f"Saved {len(rates)} FX rows to {args.output}")


if __name__ == "__main__":
    main()
