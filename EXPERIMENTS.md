# Momentum experiment comparison

Run the four fixed comparison cases with:

```text
uv run quant-trader experiment compare
```

Optional overrides are `--start-date`, `--end-date`, `--initial-capital`, and
`--output-root`. Dates are inclusive. Fees, fixed order fees, and slippage remain
fixed at 0.10%, EUR 1.00, and 0.05%, respectively.

The matrix is:

| ID | Signals | Rebalancing | Top N | Sizing |
| --- | --- | --- | ---: | --- |
| ABC-W5 | A+B+C | Weekly | 5 | Inverse volatility |
| ABC-M5 | A+B+C | Monthly | 5 | Inverse volatility |
| ABC-M10 | A+B+C | Monthly | 10 | Inverse volatility |
| AB-M10 | A+B | Monthly | 10 | Equal weight |

Each run gets a unique directory under `results/experiments/` by default. It
contains `config.json` (restorable with `load_experiment_config`), `results.csv`,
`equity_curves.csv`, `report.html`, `report.md`, and six interactive Plotly HTML
charts. A failed variant is recorded with `Status=error` and its exception; the
CLI exits nonzero after saving the partial report. Missing metrics remain blank,
not zero.

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