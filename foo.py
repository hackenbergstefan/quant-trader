import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf


def download_stock_data(tickers):
    raw = yf.download(
        tickers=list(tickers),
        start="2010-01-01",
        end="2026-11-08",
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    return raw


def save_stock_data(raw, output_dir=Path("data")):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(raw.columns, pd.MultiIndex):
        return

    for ticker in raw.columns.get_level_values(1).unique():
        close_key = ("Close", ticker)
        volume_key = ("Volume", ticker)
        if close_key not in raw.columns or volume_key not in raw.columns:
            continue
        adjusted_close_key = ("Adj Close", ticker)
        data = pd.DataFrame(
            {
                "Close": raw[adjusted_close_key]
                if adjusted_close_key in raw.columns
                else raw[close_key],
                "Volume": raw[volume_key],
                "Turnover": raw[close_key] * raw[volume_key],
                "High": raw[("High", ticker)],
                "Low": raw[("Low", ticker)],
                "Open": raw[("Open", ticker)],
                "Raw Close": raw[close_key],
            }
        )
        data.to_csv(output_dir / f"{ticker}.csv", index_label="Date")


def ticker_info(isin):
    infofile = Path(f"data/{isin}.yaml")
    if infofile.exists():
        return
    try:
        ticker = yf.Ticker(isin)
        info = ticker.info

        with infofile.open("w") as f:
            yaml.dump(info, f, allow_unicode=True)
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-prices",
        action="store_true",
        default=False,
        help="Historische Close- und Volume-Daten erneut herunterladen.",
    )
    args = parser.parse_args()
    # data = pd.read_csv("stoxx_europe_600.csv", dtype=str, keep_default_na=False)
    data = pd.read_csv("sp_500.csv", dtype=str, keep_default_na=False)
    if args.download_prices:
        for i in range(0, len(data), 10):
            tickers = data["ISIN"].iloc[i : i + 10]
            raw = download_stock_data(tickers)
            save_stock_data(raw)

    else:
        with multiprocessing.Pool() as pool:
            pool.map(ticker_info, data["ISIN"])


if __name__ == "__main__":
    main()
