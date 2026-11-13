import logging
from pathlib import Path

from pytrading import BrokerConfig, Portfolio
from pytrading.data_helper import Universe
from pytrading.strategies.momentum import MomentumConfig, variant_experiment

ROOT = Path(__file__).resolve().parent
START_DATE = "2010-01-01"
END_DATE = "2017-01-01"
RESULTS_FILE = ROOT / "experiment_results.csv"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
    )
    universe = Universe()
    print(
        "Signal D ist vorläufig: Es werden keine Benchmark-Ersatzdaten verwendet; "
        "ohne historische Benchmarks bleibt das Marktregime unbekannt."
    )

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
        benchmarks={},
    )
    config = MomentumConfig(
        topn=5,
        min_avg_daily_turnover=50_000.0,
        liquidity_lookback=20,
        momentum_20_weight=0.2,
        momentum_60_weight=0.3,
        momentum_120_weight=0.5,
        rebalance_frequency="weekly",
        max_position_weight=0.2,
        position_sizing="inverse_volatility",
        bullish_exposure=1.0,
        neutral_exposure=0.5,
        bearish_exposure=0.0,
    )

    results = variant_experiment(portfolio, config)
    results["Signal D Status"] = results["Variant"].map(
        lambda variant: (
            "vorläufig: keine historischen Benchmarks; Regime unbekannt"
            if "D" in variant
            else "nicht verwendet"
        )
    )
    results = results.rename(columns={"Variant": "Signalvariante"})
    results.to_csv(RESULTS_FILE, index=False)
    columns = [
        "Signalvariante",
        "Signals",
        "Start Date",
        "End Date",
        "rebalance_frequency",
        "topn",
        "momentum_20_weight",
        "momentum_60_weight",
        "momentum_120_weight",
        "min_avg_daily_turnover",
        "max_position_weight",
        "Initial Cash",
        "Fees",
        "Fixed Fees",
        "Slippage",
        "Universe Fingerprint",
        "CAGR",
        "Total Return",
        "Gross Total Return",
        "Maximum Drawdown",
        "Sharpe",
        "Buy Orders",
        "Sell Orders",
        "Annual Transaction Costs",
        "Annual Portfolio Turnover",
        "Average Holding Period (sessions)",
        "Average Exposure",
        "Maximum Exposure",
        "Transaction Costs",
        "Signal D Status",
    ]
    print("\nVariantenvergleich (identische Daten, Zeitraum, Kosten und Config):")
    print(results.loc[:, columns].to_string(index=False, float_format="{:.3f}".format))
    print(f"\nErgebnisdatei: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
