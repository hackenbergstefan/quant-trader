# Momentum strategy comparison

Run: `compare-20261113T163230Z-4f286dac19`
Completed: 4/4 variants

## Interpretation
- Highest net total return: ABC-M5 (268.82%).
- Smallest maximum drawdown: ABC-M10 (-25.41%).
- Lowest annualized two-sided turnover: AB-M10 (15.69x).
- Highest Sharpe ratio: AB-M10 (0.848); winner is not selected by CAGR alone.
- ABC-W5: gross/net return gap 206.56 percentage points.
- ABC-M5: gross/net return gap 134.65 percentage points.
- ABC-M10: gross/net return gap 154.66 percentage points.
- AB-M10: gross/net return gap 156.88 percentage points.
- Monthly vs weekly ABC-W5 net return difference: +236.68 percentage points; compare Sharpe, drawdown, turnover and costs alongside return.
- ABC-M10 vs ABC-M5: net return difference -147.88 percentage points, with Sharpe and risk metrics reported separately.

## Limitations
Instrument files have no historical index-membership dates; results may contain survivorship bias and are not survivorship-bias-free.

## Files
- `results.csv`
- `equity_curves.csv` (when at least one variant succeeds)
- `config.json`
- `equity_net.html`
- `equity_gross.html`
- `drawdowns.html`
- `annual_net_returns.html`
- `annual_turnover.html`
- `annual_transaction_costs.html`
