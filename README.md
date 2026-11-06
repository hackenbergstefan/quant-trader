# Quant Trader

A modular Python research and backtesting foundation for European equities. Phase 1 is research-only: it has no broker connection or order-submission code.

## Install

Requires Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
uv run quant-trader --help
```

The checked-in `uv.lock` pins the resolved dependency set. Run commands from the project root unless a different configuration directory is passed.

## Quick Start

The synthetic data generator is deterministic and offline. Its output is for pipeline validation only, not a market-performance claim.

```bash
uv run quant-trader universe
uv run quant-trader backtest run --data-source synthetic
uv run quant-trader experiment run --data-source synthetic --report
uv run quant-trader experiments holdout --data-source synthetic
uv run quant-trader experiments walk-forward --data-source synthetic
uv run quant-trader experiments sensitivity --data-source synthetic
```

The experiment command writes the comparison summary and equity curves to `data/backtests/experiment/`, with full per-variant JSON and Markdown reports, CSV files, Plotly HTML charts, and a `report.ipynb` notebook under `reports/`. Each notebook embeds interactive chart outputs and the complete CSV tables. `--output-dir` changes the root output directory. The first chart writes a local Plotly JavaScript bundle, so reports can be viewed without a CDN connection. `backtest compare` remains available for comparison-only output.

Fetch provider data and run against the DuckDB store:

```bash
uv run quant-trader data update --start 2010-01-01 --end 2024-12-31
uv run quant-trader data validate
uv run quant-trader backtest run --strategy adaptive_momentum --data-source stored
uv run quant-trader experiment run --data-source stored --report
```

`data update --instrument-ids SAP.DE,ASML.AS,NESN.SW,SHEL.L,NOVO-B.CO,EQNR.OL` fetches a bounded configured subset plus the benchmark. `data update` returns a non-zero status if any selected instrument fails. It does not silently substitute another symbol. `backtest run` accepts `--start`, `--end`, `--capital`, `--allocation-method`, and `--output-dir`; configuration is otherwise read from YAML. `data validate` compares stored sessions with each instrument's configured exchange calendar.

## Layout

```text
config/                     Strategy, risk, execution, data, and 117-name universe settings
src/quant_trader/domain/     Instruments, corporate actions, portfolio and transaction types
src/quant_trader/data/       Yahoo adapter, normalization, calendars, validation, DuckDB, synthetic data
src/quant_trader/features/   Momentum, trend, ATR, volatility, and EUR liquidity features
src/quant_trader/strategies/ Cross-sectional momentum and market regime signals
src/quant_trader/portfolio/  Allocation and entry risk constraints
src/quant_trader/backtesting/Next-open daily-bar simulator, benchmarks, and metrics
src/quant_trader/experiments/Variants, sensitivity, holdout, and walk-forward workflows
src/quant_trader/reporting/  Reproducible JSON/Markdown reports and Plotly charts
tests/                       Focused unit and synthetic integration behavior tests
```

## Data And Provider Limits

`MarketDataProvider`, `HistoricalPriceProvider`, `InstrumentRepository`, and `TradingCalendar` are replaceable protocols. `YahooFinanceProvider` is the initial usable adapter. It retrieves daily OHLCV, adjusted close, dividends, stock splits, and FX reference rates through the `yfinance` library.

Yahoo Finance is a convenient prototype source, not a professional market-data contract. Availability, history, symbol mapping, rate limits, corporate-action corrections, and API behavior are not guaranteed. `yfinance` is an unofficial wrapper; users must review Yahoo's terms and applicable data licensing before use, especially for redistribution or commercial activity. No point-in-time constituent history is provided here.

The configurable universe is a curated present-day large-cap seed (117 symbols), not a survivorship-free historical universe. Some symbols may have changed, delisted, or lack full requested history. Failed or unavailable symbols are reported, not replaced. Results on this universe must not be presented as unbiased historical performance for the European equity market. The synthetic source is separately labelled in reports and must never be treated as observed market data.

Bars store split-restored historical OHLCV for simulated fills and Yahoo's adjusted close separately. Yahoo Finance's historical `Close`/volume are split-back-adjusted even when automatic dividend adjustment is disabled; the adapter reverses recorded split ratios to restore each bar's contemporaneous share units. Indicators use adjusted close when the series is complete; ATR is converted back to the corresponding fill-price units for stop sizing. Corporate actions are retained separately and applied to held positions. Liquidity uses close times volume converted to EUR using the latest prior-session FX observation, avoiding same-day FX-close leakage. Currency metadata and the EUR base currency are explicit.

Exchange session dates are provider daily-bar dates normalized to timezone-naive dates. The calendar module uses named exchange calendars; it never infers a calendar from weekdays. Missing-session checks are based on the configured calendar and should be reviewed around IPOs, suspensions, and provider omissions. Yahoo's daily bars do not provide point-in-time fundamentals or point-in-time index constituents.

## Strategy And Simulation

The default `adaptive_momentum` strategy uses the specified 20/60/120-session weighted momentum, SMA 50/200, ATR 14, 20/60-session realized volatility, and 20-session average daily traded value. It selects up to five eligible instruments after each daily close. The EURO STOXX 50 regime uses a configurable neutral band around SMA 200; unknown reference data maps to zero exposure, never bullish by default.

Signals from close $T$ are queued for each instrument's next observed session open. Daily bars are asynchronous across exchanges, so cross-sectional ranks compare the bars available on a date; this is a daily research approximation, not an intraday synchronized market snapshot. Existing positions can exit on a zero target or a stop. A gap through a stop fills at that session's open; an intraday low touching a stop fills at the stop threshold. Positions are closed at the last available close and charged exit costs.

The configured cost model is a linear basis-point estimate applied to notional: commission, exchange fees, spread, and slippage are separate components and summed once. It is not a broker fee schedule or order-book simulation. Dividends are credited on ex-date using the latest prior-session EUR FX rate; split fractions are rounded down and settled as cash-in-lieu at the ex-date open. Taxes are not modeled; reports are tax-neutral.

Entry size is constrained by risk budget, target allocation, the EUR 1,000 per-position cap, total exposure, and available cash. The risk budget and stops do not guarantee a 50% maximum drawdown: gaps, suspensions, data errors, and correlated holdings can exceed intended risk. There is no portfolio-level 50% drawdown circuit breaker.

Variants in `experiment run` and `backtest compare` are A (EURO STOXX 50 adjusted-close buy-and-hold proxy), B (momentum), C (momentum plus trend), D (trend with volatility-adjusted weights), E (full regime-adaptive strategy), and cash. All use the same supplied period and cost configuration. The benchmark is a price-series proxy, not a tradable ETF replication, and its adjusted close is subject to the same provider limitations.

Walk-forward uses expanding chronological test windows with the fixed, configured rule set; it does not fit parameters on its nominal training window. Holdout defaults to 2010-2022 and 2023-2024. Sensitivity runs vary neutral-band/stop settings and multiply all costs. The implementation reports actual available coverage and marks missing holdout periods unavailable; it never fabricates market history. These tools support research, not a guarantee of statistical robustness.

## Reproducibility

Each backtest report records the strategy version, validated configuration and SHA-256, normalized input-data SHA-256, data source, actual date/instrument/bar coverage, execution assumptions, and UTC creation time. The raw provider payload and normalized bars are stored separately in DuckDB. Given the same normalized bars, configuration, and package version, runs are deterministic.

DuckDB's default path is `data/market.duckdb`. Reports are ignored by Git under `data/backtests/`; do not commit licensed provider data without verifying redistribution rights.

## Limitations And Next Steps

- No point-in-time universe, delisted-security reconstruction, or survivorship-bias correction.
- Yahoo data coverage and licensing are not adequate evidence for production or commercial research without review.
- No trading halt, auction, partial-fill, market-impact, borrow, short-selling, or exchange-specific order-rule model.
- Corporate-action and FX quality depends on the provider; cash-in-lieu uses the ex-date open as an explicit approximation.
- No taxes, funding yield, cash interest, or portfolio-level hard drawdown stop.
- Walk-forward windows are evaluation windows only; no parameter selection or statistical significance correction is performed.

Recommended next steps are a licensed, point-in-time-capable European data source; historical constituent/delisted coverage; independently reconciled corporate actions and FX; and validation of open-auction, halts, fees, and cash-in-lieu assumptions before interpreting results beyond research prototyping.