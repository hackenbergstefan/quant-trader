from pathlib import Path

import pandas as pd

from pytrading import BrokerConfig, Portfolio
from pytrading.data_helper import Universe
from pytrading.strategies.momentum import MomentumConfig, variant_experiment

ROOT = Path(__file__).resolve().parent
START_DATE = "2010-01-01"
END_DATE = "2016-01-01"


def _constituent_benchmark(
    close: pd.DataFrame, members_file: Path, label: str
) -> tuple[pd.Series, int]:
    """Build an equal-weight proxy from the locally available current members."""
    members = pd.read_csv(members_file)["ISIN"].dropna().astype(str).str.strip()
    available = [
        isin for isin in members if isin in close.columns and close[isin].notna().any()
    ]
    if not available:
        raise ValueError(f"Keine lokalen Kursdaten für den {label}-Benchmark gefunden.")

    daily_returns = close.loc[:, available].pct_change(fill_method=None)
    benchmark_returns = daily_returns.mean(axis=1).fillna(0.0)
    benchmark = (1 + benchmark_returns).cumprod()
    benchmark.name = label
    return benchmark, len(available)


def main() -> None:
    universe = Universe()
    close = universe.close()
    benchmarks = {}
    for region, filename in (
        ("EUROPE", "stoxx_europe_600.csv"),
        ("USA", "sp_500.csv"),
    ):
        benchmarks[region], member_count = _constituent_benchmark(
            close, ROOT / filename, region
        )
        print(f"{region}-Benchmark: {member_count} lokale Indexmitglieder")

    portfolio = Portfolio(
        universe=universe,
        start_date=START_DATE,
        end_date=END_DATE,
        broker=BrokerConfig(
            initial_cash=3000.0,
            fees=0.001,
            fixed_fees=1.0,
            slippage=0.0005,
        ),
        benchmarks=benchmarks,
    )
    config = MomentumConfig(
        topn=10,
        min_avg_daily_turnover=1_000_000.0,
    )

    results = variant_experiment(portfolio, config)
    columns = [
        "Variant",
        "Signals",
        "CAGR",
        "Total Return",
        "Maximum Drawdown",
        "Sharpe",
        "Average Exposure",
        "Maximum Exposure",
        "Transaction Costs",
    ]
    print("\nVariantenvergleich (identische Daten, Zeitraum, Kosten und Config):")
    print(results.loc[:, columns].to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
