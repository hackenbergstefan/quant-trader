from pathlib import Path

import yfinance as yf
import yaml
import pandas as pd
import multiprocessing


def load_csv():
    data = pd.read_csv("trade_republic_aktien.csv", dtype=str, keep_default_na=False)
    return data


def download_stock_data(tickers):
    raw = yf.download(
        tickers=list(tickers),
        start="2010-01-01",
        end="2026-11-08",
        interval="1d",
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    return raw


def save_stock_data(raw):
    for col in raw.columns.levels[1]:
        df = raw[("Close", col)].copy()
        df.to_csv(f"data/{col}.csv")


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
    data = load_csv()
    for i in range(0, len(data), 10):
        pass
        # print(i)
        # raw = download_stock_data(data["isin"][i : i + 10])
        # save_stock_data(raw)

    with multiprocessing.Pool() as pool:
        pool.map(ticker_info, data["isin"])


if __name__ == "__main__":
    main()
