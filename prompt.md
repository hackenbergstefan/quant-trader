# Engineering Task: Implement Phase 1 of an Algorithmic Trading System

## Role

Act as a senior quantitative developer and Python software architect with expertise in:

- Algorithmic trading and quantitative finance
- Financial time series and market data
- Backtesting and statistical validation
- Python software architecture
- Reproducible scientific experiments
- Automated testing and data engineering

Your task is to implement Phase 1 of a modular algorithmic trading system for European equities.

Do not merely produce an architecture proposal or illustrative code. Implement a complete, executable project with tests, documentation and a working initial backtest.

Make reasonable engineering decisions independently. Document important assumptions and trade-offs. Do not block implementation with unnecessary clarification questions.

---

# 1. Project objectives

Build the foundation of an algorithmic trading platform that will eventually support automated analysis and trading of European equities.

The long-term system requirements are:

| Property | Requirement |
|---|---|
| Market | European equities |
| Trading style | Adaptive algorithmic trading |
| Initial strategy | Momentum + Trend Following |
| Trading frequency | Adaptive, initially daily |
| Capital | EUR 5,000 |
| Risk tolerance | Maximum tolerated portfolio loss approximately 50% |
| Optimization objective | Maximize absolute net return |
| Execution | Manual order approval in later phases |
| Implementation language | Python |

Phase 1 is exclusively concerned with:

1. Market data acquisition and normalization.
2. Historical data storage.
3. Feature engineering.
4. Strategy implementation.
5. Portfolio simulation.
6. Backtesting.
7. Performance analysis.

Do not implement live trading, broker connectivity or actual order execution in this phase.

The architecture must nevertheless allow these capabilities to be added later without redesigning the strategy implementation.

# 2. Engineering principles

Follow these principles throughout the project:

- Modular architecture with clearly defined interfaces.
- Separation of domain logic, data access and strategy implementation.
- Deterministic and reproducible backtests.
- Strict prevention of look-ahead bias.
- Explicit handling of missing data and corporate actions.
- No implicit assumptions about trading calendars.
- No hidden global state.
- Type hints for public interfaces.
- Meaningful logging and structured error handling.
- Comprehensive unit and integration tests.
- Configuration instead of hardcoded strategy parameters.
- Prefer simple, understandable implementations over premature abstraction.

Use a modular Python monolith. Do not introduce microservices.

Use current stable versions of dependencies and specify them in pyproject.toml.

# 3. Technology stack

Use the following initial technology choices:

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Dependency management | uv |
| Data processing | pandas, NumPy |
| Data storage | DuckDB |
| Backtesting | vectorbt, or a custom event-driven engine if required for correctness |
| Configuration | Pydantic / YAML |
| Testing | pytest |
| Logging | Python logging |
| Packaging | pyproject.toml |
| Visualization | plotly |
| CLI | Typer |

You may deviate from these choices when there is a compelling technical reason. Explain the deviation.

Avoid unnecessary dependencies.

# 4. Market data architecture

Implement a provider-independent market data interface.

Required interfaces:

- MarketDataProvider
- HistoricalPriceProvider
- InstrumentRepository
- TradingCalendar

The system must support daily OHLCV data.

Required fields:

- instrument_id
- symbol
- exchange
- timestamp
- open
- high
- low
- close
- adjusted_close, if available
- volume
- currency

Data requirements:

- Historical prices must account for stock splits and dividends.
- Avoid mixing adjusted and unadjusted price series.
- Preserve raw source data separately from normalized data.
- Detect duplicate observations.
- Validate OHLC relationships.
- Detect missing trading sessions.
- Handle different exchange calendars.
- Normalize timestamps and currencies.
- Record data source and retrieval timestamps.

Implement at least one usable historical data provider.

Evaluate available providers for European equity coverage, historical depth, API limitations, licensing and reliability.

A free provider may be used for the initial prototype, but its limitations must be documented.

Do not assume that a data provider offers point-in-time historical index constituents or survivorship-free historical universes.

Make the provider replaceable without modifying strategy code.

# 5. Initial trading universe

Start with approximately 100–200 liquid European equities.

Initial geographical coverage:

- Germany
- France
- Netherlands
- Switzerland
- United Kingdom
- Nordic countries

Prefer liquid large-cap equities.

The universe must be configurable independently from the strategy.

Implement:

- Instrument metadata
- Exchange identification
- Currency
- Trading calendar association
- Minimum liquidity filters
- Historical instrument availability

Important:

Prevent survivorship bias wherever possible.

If historical point-in-time constituents are unavailable, explicitly document this limitation and distinguish between:

1. Historical performance on today's surviving universe.
2. Performance using a genuinely historical point-in-time universe.

Never present the first as an unbiased historical market-wide result.

Do not silently replace delisted or unavailable instruments.

# 6. Initial strategy: Adaptive European Momentum v0.1

Implement the first strategy as a transparent, deterministic, rule-based strategy.

No machine learning in this phase.

## 6.1 Indicators

Calculate the following indicators for each instrument:

- Momentum 20 trading days
- Momentum 60 trading days
- Momentum 120 trading days
- SMA 50
- SMA 200
- ATR 14
- Average daily traded value over 20 days
- Realized volatility over 20 and 60 days

Momentum definitions:

M20 = Close(t) / Close(t-20) - 1

M60 = Close(t) / Close(t-60) - 1

M120 = Close(t) / Close(t-120) - 1

Composite momentum score:

Score = 0.2 * M20 + 0.3 * M60 + 0.5 * M120

All indicator calculations must be tested for correct indexing and warm-up periods.

## 6.2 Eligibility filter

An instrument is eligible if:

- Momentum score is positive.
- Close is above SMA 50.
- SMA 50 is above SMA 200.
- Average daily traded value exceeds EUR 1,000,000.
- Sufficient historical data is available.

Liquidity must be evaluated in EUR.

## 6.3 Cross-sectional ranking

On each trading day:

1. Evaluate all eligible instruments.
2. Calculate their momentum scores.
3. Rank instruments in descending order.
4. Select up to five instruments.
5. Avoid unnecessary portfolio turnover.

The ranking must be calculated using information available at the decision timestamp only.

## 6.4 Market regime

Use EURO STOXX 50 as the initial market reference.

Implement three regimes:

- BULLISH: Index above SMA 200.
- NEUTRAL: Index close to SMA 200.
- BEARISH: Index below SMA 200.

Use configurable thresholds to define the neutral zone.

Initial target exposure:

- Bullish: up to 100%.
- Neutral: up to 50%.
- Bearish: 0% new long exposure.

These are initial hypotheses, not optimized values.

Existing positions must be managed according to explicit exit rules.

If the reference index data is unavailable, do not silently classify the market as bullish.

## 6.5 Exit rules

Implement configurable exit conditions:

- Close below SMA 50.
- Momentum score becomes negative.
- Instrument is no longer among the selected candidates.
- Stop-loss threshold is reached.

Initial stop-loss:

2 * ATR(14) below entry price.

Document stop execution assumptions carefully.

Do not assume a stop order can always execute at the stop price.

# 7. Portfolio construction

Initial configuration:

- Starting capital: EUR 5,000.
- Maximum positions: 5.
- Maximum position value: EUR 1,000.
- Maximum portfolio exposure: 100% before market regime adjustment.
- Risk budget per position: 1% of portfolio equity.
- Initial stop distance: 2 * ATR.

Position sizing:

Shares = min(
    Risk Budget / Stop Distance,
    Maximum Position Value / Entry Price
)

Round share quantities down to whole shares.

Respect the available cash balance.

Do not allow accidental leverage or negative cash.

Support configurable allocation methods, initially:

- Equal weight.
- Volatility-adjusted weight.
- Risk-budget-based weight.

Use equal weighting as the initial baseline.

# 8. Backtesting engine

The backtesting engine is a critical component.

Prioritize correctness and transparency over performance.

Requirements:

- Daily portfolio valuation.
- Cash accounting.
- Position accounting.
- Transaction history.
- Entry and exit prices.
- Trading costs.
- Slippage.
- Spread assumptions.
- Dividends.
- Stock splits.
- Trading calendar handling.
- Unfilled order simulation where applicable.

Prevent look-ahead bias.

A signal calculated using closing prices of day T must not execute at that same closing price.

For the initial implementation, calculate signals after market close and simulate execution at the next trading session's open, with configurable execution costs.

Use explicit, configurable transaction cost assumptions.

Document the difference between broker commissions, exchange fees, spread and slippage.

Do not double-count these costs.

The engine must support deterministic runs with fixed configuration and identical input data.

# 9. Benchmarks and performance metrics

Compare all strategy variants against:

- EURO STOXX 50 buy and hold.
- A simple cash benchmark.

Calculate at least:

- Total return
- CAGR
- Maximum drawdown
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Profit factor
- Win rate
- Average win
- Average loss
- Number of trades
- Average holding period
- Annual turnover
- Average capital exposure
- Transaction costs

Clearly distinguish gross and net performance.

Use EUR as portfolio base currency.

Include a separate tax-neutral performance report. German tax treatment can be implemented in a later phase.

# 10. Strategy experiments

Implement the following variants:

A: Buy and hold EURO STOXX 50.

B: Simple momentum ranking.

C: Momentum + trend filter.

D: Momentum + trend filter + volatility-adjusted position sizing.

E: Full adaptive strategy including market regime filtering.

All variants must use the same data period and comparable cost assumptions.

Generate a comparison report.

Do not automatically declare the highest-returning strategy the winner.

Highlight differences in risk, drawdown, turnover and robustness.

# 11. Validation and bias prevention

Implement the following validation methodology:

- In-sample period: 2010–2022, subject to data availability.
- Out-of-sample period: 2023–2024, subject to data availability.
- Walk-forward validation with configurable windows.
- Parameter sensitivity analysis.
- Transaction cost sensitivity analysis.

Prevent:

- Look-ahead bias.
- Survivorship bias where historical data permits.
- Data leakage.
- Incorrect corporate-action handling.
- Unrealistic execution assumptions.
- Overfitting through excessive parameter optimization.

The test framework must make it possible to verify that a strategy does not use future information.

If historical data does not cover the specified periods, report the actual coverage and do not fabricate results.

# 12. Project structure

Use the following proposed structure:

quant-trader/
    pyproject.toml
    README.md
    config/
        strategy.yaml
        universe.yaml
        risk.yaml
        data.yaml

    src/
        quant_trader/
            domain/
                models.py
                portfolio.py
                orders.py

            data/
                providers.py
                market_data.py
                instruments.py
                corporate_actions.py
                calendars.py

            features/
                momentum.py
                trend.py
                volatility.py
                liquidity.py

            strategies/
                base.py
                adaptive_momentum.py

            portfolio/
                allocation.py
                risk_manager.py

            backtesting/
                engine.py
                execution.py
                metrics.py
                benchmark.py

            experiments/
                runner.py
                comparison.py
                walk_forward.py

            reporting/
                reports.py
                charts.py

            cli.py

    tests/
        unit/
        integration/
        backtesting/

    data/
        raw/
        processed/
        backtests/

Adapt the structure if a simpler design is technically superior, but preserve the separation of responsibilities.

# 13. CLI

Provide commands similar to:

    quant-trader data update

    quant-trader data validate

    quant-trader universe show

    quant-trader backtest run --strategy adaptive_momentum

    quant-trader backtest compare

    quant-trader report generate

    quant-trader experiments walk-forward

The CLI must be usable without editing Python source code.

Configuration should be passed through YAML files and command-line overrides.

# 14. Reporting

Generate:

1. A machine-readable JSON report.
2. A human-readable Markdown report.
3. Equity curve chart.
4. Drawdown chart.
5. Monthly returns chart.
6. Strategy comparison table.
7. Trade history as CSV.

Each report must include:

- Strategy version.
- Configuration hash.
- Data source.
- Data coverage.
- Execution assumptions.
- Backtest period.
- Creation timestamp.

A backtest must be reproducible from its recorded configuration and input data.

# 15. Testing requirements

Implement unit tests for:

- Momentum calculations.
- SMA calculations.
- ATR calculations.
- Eligibility filters.
- Cross-sectional ranking.
- Market regime classification.
- Position sizing.
- Stop-loss calculations.
- Cash accounting.
- Corporate actions.
- Transaction costs.
- Execution timing.

Implement integration tests using synthetic datasets with known expected results.

Explicitly test:

- No same-day close execution after close-based signals.
- No future data leakage.
- No negative cash.
- No position exceeding configured limits.
- Correct handling of missing observations.
- Correct treatment of stock splits.
- Deterministic backtest results.

All tests must pass.

# 16. Development workflow

Work autonomously in the existing repository or create the project in the current working directory if no repository exists.

Proceed in this order:

1. Inspect the existing environment and repository.
2. Establish the project structure.
3. Implement domain models and interfaces.
4. Implement market data provider.
5. Implement data validation and persistence.
6. Implement feature calculations.
7. Implement strategy.
8. Implement portfolio construction.
9. Implement backtesting.
10. Implement performance metrics.
11. Implement experiment comparison.
12. Implement CLI and reporting.
13. Write and run tests.
14. Fix all discovered issues.
15. Update README with installation and usage instructions.

Do not stop after producing a plan.

Do not replace actual implementation with pseudocode.

Do not fabricate successful backtest results if data retrieval fails.

If external API access or credentials are unavailable, implement the provider interface and a deterministic synthetic-data integration test so that the entire pipeline can still be executed.

# 17. Definition of Done

Phase 1 is complete when:

- The project installs successfully.
- The CLI works.
- Historical data can be loaded and normalized.
- Data validation works.
- All required indicators are calculated correctly.
- The adaptive momentum strategy produces deterministic signals.
- Portfolio construction respects capital and risk constraints.
- Backtests can be executed.
- Transaction costs are accounted for.
- Benchmark comparisons work.
- Reports are generated.
- Unit and integration tests pass.
- No known look-ahead errors remain in the tested execution path.
- All material limitations are documented.

# 18. Final delivery

At completion, provide:

1. Summary of implemented components.
2. Project structure.
3. Important architecture decisions.
4. Installation instructions.
5. Example CLI commands.
6. Test results.
7. Actual backtest results, if historical data was successfully retrieved.
8. Known limitations and unresolved risks.
9. Recommended next engineering steps.

Be critical of your own implementation.

Correctness, reproducibility and data integrity take priority over impressive-looking performance numbers.

The result should be a maintainable foundation for a serious quantitative trading platform, not a demonstration script.