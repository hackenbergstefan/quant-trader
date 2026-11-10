import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml


def load_csv(file):
    tickerinfo = yaml.safe_load(file.open())
    if tickerinfo["currency"] != "EUR":
        return None
    if tickerinfo["quoteType"] != "EQUITY":
        return None
    csvfile = file.with_suffix(".csv")
    if not csvfile.exists():
        return None
    df = pd.read_csv(
        csvfile,
        skiprows=2,
        parse_dates=["Date"],
        index_col="Date",
    )
    df.columns = [tickerinfo["symbol"]]
    return df


def load_csvs():
    if Path("close.pkl").exists():
        return pd.read_pickle(Path("close.pkl"))
    data = pd.DataFrame(
        index=pd.date_range(start="2010-01-01", end="2026-11-06", freq="D")
    )
    with multiprocessing.Pool() as pool:
        dfs = pool.map(load_csv, Path("data").glob("*.yaml"))
    data = pd.concat([data, *dfs], axis=1, sort=False)
    data = data.dropna(axis=0, how="all")
    data = data.dropna(axis=1, how="all")
    data.to_pickle(Path("close.pkl"))
    return data


def momentum_portfolio(
    close: pd.DataFrame,
    start_date: str,
    end_date: str,
    lookback: int = 126,
    topn: int = 10,
    initial_cash: float = 1000.0,
    fees: float = 0.001,
    fixed_fees: float = 1.0,
    slippage: float = 0.0005,
):
    # Setup tables and momentum
    lookback_prices = close[(close.index >= start_date) & (close.index <= end_date)]
    lookback_prices = lookback_prices.dropna(axis=0, how="all")
    lookback_prices = lookback_prices.dropna(axis=1, how="all")
    momentum = close.pct_change(lookback)

    # Determine the signal dates and execution dates
    first_date = lookback_prices.index[0]
    previous_dates = close.index[close.index < first_date]
    events = []
    if len(previous_dates) > 0:
        events.append((first_date, previous_dates[-1]))

    month_ends = (
        lookback_prices.groupby(lookback_prices.index.to_period("M")).tail(1).index
    )

    for signal_date in month_ends:
        position = lookback_prices.index.get_loc(signal_date)
        if position + 1 >= len(lookback_prices.index):
            continue
        execution_date = lookback_prices.index[position + 1]
        if execution_date >= first_date:
            events.append((execution_date, signal_date))

    # Calculate the weights for each execution date based on the momentum scores
    weights = pd.DataFrame(
        np.nan,
        index=lookback_prices.index,
        columns=lookback_prices.columns,
        dtype=float,
    )
    for execution_date, signal_date in events:
        scores = momentum.loc[signal_date].dropna()
        scores = scores.sort_values(ascending=False)
        execution_prices = close.loc[execution_date].reindex(weights.columns)
        valid_prices = execution_prices[
            np.isfinite(execution_prices) & (execution_prices > 0)
        ]
        candidates = valid_prices.reindex(scores.index).dropna()
        top = candidates.head(topn).index.tolist()
        if len(top) < topn:
            continue
        weights.loc[execution_date, valid_prices.index] = 0.0
        weights.loc[execution_date, top] = 1.0 / len(top)

    return vbt.Portfolio.from_orders(
        close=lookback_prices,
        size=weights,
        size_type="targetpercent",
        direction="longonly",
        init_cash=initial_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        allow_partial=True,
        freq="1D",
        log=True,
    )
