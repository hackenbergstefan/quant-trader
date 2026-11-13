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
`equity_curves.csv`, `report.html`, `report.md`, and seven interactive Plotly
HTML charts, including `position_weights.html`. A failed variant is recorded
with `Status=error` and its exception; the CLI exits nonzero after saving the
partial report. Missing metrics remain blank, not zero.

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
Annual transaction costs are grouped by order year. Holding periods use FIFO
matched sale portions; open lots are excluded from completed trade holding
periods and reported separately.

AB positions are equal weighted; ABC positions use inverse volatility from an
equal blend of annualized 20- and 60-session realized volatility. Both sizing
rules enforce the 20% per-position cap and 100% total-exposure cap. If fewer
than five eligible positions exist, unused capital remains cash. The results
include mean, median, and maximum actual daily position weights, mean held-name
volatility, weight-weighted constituent volatility, invested-weight HHI, and
the share of sessions with a position at or above the cap. The
weight-weighted constituent volatility is not covariance-aware portfolio
volatility; compare it with realized volatility of portfolio returns to assess
diversification and realized risk.

The report compares ABC with AB at Top 5 and Top 10 using net return, Sharpe,
realized portfolio volatility, weight-weighted constituent volatility,
turnover, and costs. It also compares Top 5 with Top 10 within each sizing rule.

The daily close series is adjusted. Raw Open, High, and Low are scaled by
`Close / Raw Close` before EUR conversion so execution and valuation use the
same corporate-action-adjusted price basis. EUR conversion uses the locally
archived FX rate carried forward from the most recent prior observation.

## Data limitations

The Universe loads the instrument files currently present in `data/`; those
files do not include historical index-membership dates. The comparison can
therefore contain survivorship bias and must not be described as
survivorship-bias-free. FX coverage and historical price data must already be
available locally; the runner does not substitute missing values or download
replacement data. Signal D and regional regime controls are excluded from this
matrix.