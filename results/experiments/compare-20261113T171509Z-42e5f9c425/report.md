# Momentum strategy comparison

Run: `compare-20261113T171509Z-42e5f9c425`
Completed: 4/4 variants

## Interpretation
- Highest net total return: ABC-M5 (268.82%).
- Smallest maximum drawdown: ABC-M10 (-25.41%).
- Lowest annualized two-sided turnover: AB-M10 (15.69x).
- Highest Sharpe ratio: AB-M10 (0.848); winner is not selected by CAGR alone.
- The 20% cap applies to rebalance target weights; market moves can push actual daily position weights above it until the next monthly rebalance.
- AB-M5: gross/net return gap 129.43 percentage points.
- ABC-M5: gross/net return gap 134.65 percentage points.
- AB-M10: gross/net return gap 156.88 percentage points.
- ABC-M10: gross/net return gap 154.66 percentage points.
- Signal C at Top 5 (ABC-M5 vs AB-M5): net return change +21.24 pp; Sharpe change +0.03; realized portfolio volatility change +0.06 pp; weight-weighted constituent volatility change +0.07 pp; annualized turnover change +0.02x; transaction cost change +34.79 EUR.
- Signal C at Top 10 (ABC-M10 vs AB-M10): net return change -82.05 pp; Sharpe change -0.19; realized portfolio volatility change -1.09 pp; weight-weighted constituent volatility change -4.02 pp; annualized turnover change +1.09x; transaction cost change +96.32 EUR.
- Top 10 vs Top 5 for AB: net return change -44.59 pp; Sharpe change +0.04; maximum drawdown change +3.37 pp; annualized turnover change -1.12x; transaction cost change +433.59 EUR.
- Top 10 vs Top 5 for ABC: net return change -147.88 pp; Sharpe change -0.18; maximum drawdown change +4.19 pp; annualized turnover change -0.05x; transaction cost change +495.12 EUR.

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
- `position_weights.html`
