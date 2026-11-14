# Momentum experiment comparison

Run the four fixed comparison cases with:

```text
uv run quant-trader experiment compare
```

Optional overrides are `--start-date`, `--end-date`, `--initial-capital`, and
`--output-root`. Dates are inclusive. Fees, fixed order fees, and slippage remain
fixed at 0.10%, EUR 1.00, and 0.05%, respectively.

## Signal C semantics audit

In the previous implementation, selecting C computed realized volatility and
excluded candidates without valid volatility. Inverse-volatility weights were
used only when C was selected and `position_sizing="inverse_volatility"`;
with C and `position_sizing="equal"`, candidates still needed volatility but
were equally weighted. Without C, candidates were equally weighted even when
`position_sizing="inverse_volatility"` was configured. Signal selection and
the separate sizing option therefore overlapped and could disagree. The
comparison implementation removes that second control: C alone means inverse
volatility weighting, and no C means equal weighting.

The matrix is:

| ID | Signals | Rebalancing | Top N | Sizing |
| --- | --- | --- | ---: | --- |
| AB-M5 | A+B | Monthly | 5 | Equal weight |
| ABC-M5 | A+B+C | Monthly | 5 | Inverse volatility |
| AB-M10 | A+B | Monthly | 10 | Equal weight |
| ABC-M10 | A+B+C | Monthly | 10 | Inverse volatility |

Each run gets a unique directory under `results/experiments/` by default. It
contains `config.json` (restorable with `load_experiment_config`), `results.csv`,
`results.json`, `metric_definitions.md`, `equity_curves.csv`, `report.html`,
`report.md`, and seven interactive Plotly HTML charts, including
`position_weights.html`. CSV, JSON, and the HTML table share the same result
rows and field names; [METRICS.md](./METRICS.md) is the canonical definition.
A failed variant is recorded with `Status=error` and its exception; the CLI
exits nonzero after saving the partial report. Missing metrics remain blank or
JSON `null`, not zero.

## Calculation conventions

Signals and target weights are prepared once per case and passed unchanged to
the net and gross VectorBT simulations. Gross uses zero fees, fixed fees, and
slippage; net uses the archived broker settings. Net portfolio value already
includes these costs, so the report does not subtract them again. Slippage cost
is the absolute filled quantity times the difference between fill and next-open
reference price. Variable and fixed fees are reported separately from the
VectorBT order records.

One-sided turnover is buy notional divided by average daily portfolio value.
Two-sided turnover is buy plus sell notional divided by average daily portfolio
value. Annualized turnover divides each period ratio by observed sessions / 252.
Annual transaction costs are grouped by order year. FIFO holding metrics count
a buy lot as completed only after all shares from that buy are sold; partial
sale allocations do not count as separate lots. Completed holding periods end
when the final remaining shares of a buy lot are sold. Mapped exchanges use
their exchange sessions; unmatched lots and positions remain open at the end.

AB positions are equal weighted; ABC positions use inverse volatility from an
equal blend of annualized 20- and 60-session realized volatility. Both sizing
rules enforce the 20% per-position cap and 100% total-exposure cap. If fewer
than five eligible positions exist, unused capital remains cash. The results include mean, median, and maximum actual position weights, mean
held-name volatility, a weight-weighted average constituent volatility,
covariance-based portfolio volatility, invested-weight HHI, and the share of
sessions with an actual position at or above the target cap. Constituent
volatility ignores correlations; covariance-based volatility uses actual
close-marked weights and the trailing 60 jointly observed daily returns
(minimum 20 observations), annualized by √252. See [METRICS.md](./METRICS.md)
for exact formulas, data sources, aggregation rules, and limitations.

The 20% cap applies to target weights at each monthly rebalance. Because
positions are not forcibly rebalanced between scheduled dates, market moves can
push actual daily position weights above 20%; the report and position chart show
that drift explicitly.

The report compares ABC with AB at Top 5 and Top 10 using net return, Sharpe,
realized portfolio volatility, covariance-based portfolio volatility,
weight-weighted constituent volatility, turnover, and costs. It also compares
Top 5 with Top 10 within each sizing rule.

The daily close series is adjusted. Raw Open, High, and Low are scaled by
`Close / Raw Close` before EUR conversion so execution and valuation use the
same corporate-action-adjusted price basis. EUR conversion uses the locally
archived FX rate carried forward from the most recent prior observation.

## Three-year rolling robustness analysis

Run the four fixed variants over yearly-starting, three-calendar-year windows
with:

```text
uv run quant-trader experiment rolling-3y
```

The default start is 2010-01-01. The local close data determines the latest
available date. Window intervals use `[start_date, end_date)` semantics; each
new window has the same configured initial cash (EUR 3,000 by default) and a
fresh VectorBT portfolio, so positions and cash are never carried between
windows. The existing causal indicator preparation can use historical prices
before a window for warm-up, but simulated prices, returns, and portfolio
state are limited to that window. No signal, sizing, rebalance, or order
execution rules are changed.

The run emits `rolling_3y_results.csv` and an equivalent
`rolling_3y_results.json`, `rolling_3y_summary.csv`,
`rolling_3y_comparison.csv`, `rolling_3y_metadata.json`, a copy of
`metric_definitions.md`, an HTML report, and four Plotly charts. Detail outputs
include both nominal exclusive window end dates and the last observed
valuation date. Windows whose nominal end is later than the available local
data are marked `incomplete`; they remain visible in detail and pairwise rows,
but are excluded from aggregate statistics and charts.
Null detail metrics remain null; `Null Metric Reasons` records the documented
undefined condition, missing calendar-year coverage, or the variant error.

The summary reports min, max, arithmetic mean, median, and sample standard
deviation for each metric across complete windows. The comparison output gives
per-window and aggregate differences for CAGR, Sharpe, maximum drawdown, and
transaction costs. A positive return/Sharpe/drawdown difference is `B - A`;
the reference variant A is considered better for those metrics when its value
is higher, and for costs when its costs are lower. Overlapping windows are
dependent, so the standard deviations describe window dispersion and are not
independent-sample estimates. The report also calls out each variant's worst
complete-window CAGR and drawdown. Exact metric and aggregation definitions
are archived with every run and refer to [METRICS.md](./METRICS.md).

## Data limitations

The Universe loads the instrument files currently present in `data/`; those
files do not include historical index-membership dates. The comparison can
therefore contain survivorship bias and must not be described as
survivorship-bias-free. FX coverage and historical price data must already be
available locally; the runner does not substitute missing values or download
replacement data. Signal D and regional regime controls are excluded from this
matrix.