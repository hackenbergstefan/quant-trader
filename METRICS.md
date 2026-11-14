# Backtest metric definitions

This is the canonical definition of the calculated fields in `results.csv`,
`results.json`, and the HTML results table. Those three exports are produced
from the same result rows. Each comparison run includes a snapshot of this
document as `metric_definitions.md`. Run configuration and identity columns
(for example `Variant`, `Start Date`, `max_position_weight`, and `Status`) are
metadata, not performance metrics.

The backtest is long-only and values holdings at adjusted EUR closes. If a
market has no close on a portfolio date, exposure calculations carry forward
that instrument's last observed close. Rates and ratios are decimal-valued
(for example, `0.10` means 10%).

| Metric | Definition | Calculation | Data Source | Aggregation | Annualization | Limitations |
| --- | --- | --- | --- | --- | --- | --- |
| Initial Capital | Starting portfolio cash. | Broker `initial_cash`. | Broker configuration. | Scalar per run. | None. | Does not include pre-run positions. |
| Final Portfolio Value | Net portfolio value at the final valuation. | Last net VectorBT value. | Net simulation. | Final date. | None. | Includes trading costs already charged by the simulation. |
| Net End Value | Alias for `Final Portfolio Value`. | Last net VectorBT value. | Net simulation. | Final date. | None. | Same value as `Final Portfolio Value`. |
| Gross Final Portfolio Value | Gross portfolio value at the final valuation. | Last gross VectorBT value. | Simulation with fees, fixed fees, and slippage set to zero. | Final date. | None. | Shares the same target-weight schedule, but gross and net order sizes may differ. |
| Gross End Value | Alias for `Gross Final Portfolio Value`. | Last gross VectorBT value. | Gross simulation. | Final date. | None. | Same value as `Gross Final Portfolio Value`. |
| Gross-Net End Value Difference | Difference between gross and net final values. | Gross end value minus net end value. | Gross and net simulations. | Final date. | None. | Not a direct sum of fees: costs can change subsequent cash and order sizes. |
| Total Return | Net total return. | Net end value / initial capital - 1. | Net simulation and broker configuration. | Whole backtest. | None. | Does not annualize. |
| Net Total Return | Alias for `Total Return`. | Net end value / initial capital - 1. | Net simulation and broker configuration. | Whole backtest. | None. | Same value as `Total Return`. |
| Gross Total Return | Gross total return. | Gross end value / initial capital - 1. | Gross simulation and broker configuration. | Whole backtest. | None. | Excludes all configured fees and slippage. |
| CAGR | Net compound annual growth rate. | `(net end value / initial capital) ** (1 / elapsed calendar years) - 1`; elapsed years are first-to-last valuation days / 365.25. | Net simulation and valuation index. | Whole backtest. | Calendar-year exponent. | Short runs use a minimum duration of 1/252 year. |
| Net CAGR | Alias for `CAGR`. | Same as `CAGR`. | Net simulation and valuation index. | Whole backtest. | Calendar-year exponent. | Same value as `CAGR`. |
| Gross CAGR | Gross compound annual growth rate. | Gross end value / initial capital compounded over the same elapsed calendar years as `CAGR`. | Gross simulation and valuation index. | Whole backtest. | Calendar-year exponent. | Excludes configured fees and slippage. |
| Maximum Drawdown | Largest peak-to-trough net value decline. | Minimum of `value / cumulative_max(value) - 1`. | Net daily portfolio values. | Worst date in the run. | None. | Reported as a non-positive decimal. |
| Sharpe | Zero-risk-free-rate Sharpe ratio for net daily returns. | Mean daily return / sample standard deviation × √252. | Net daily portfolio values. | Whole run's daily returns. | 252 sessions/year. | Undefined (NaN) when return volatility is zero or unavailable. |
| Sharpe Ratio | Parameter-sweep name for `Sharpe`. | Mean daily return / sample standard deviation × √252. | Net daily portfolio values. | Whole run's daily returns. | 252 sessions/year. | Same definition as `Sharpe`; the VectorBT-native frequency annualization is not used. |
| Sortino | Zero minimum-acceptable-return Sortino ratio. | Mean net daily return / RMS of negative daily returns × √252. | Net daily portfolio values. | Whole run's daily returns. | 252 sessions/year. | Undefined (NaN) when downside deviation is zero or unavailable. |
| Sortino Ratio | Rolling-report name for `Sortino`. | Same as `Sortino`. | Net daily portfolio values. | Whole run's daily returns. | 252 sessions/year. | Alias; no separate definition. |
| Calmar | Net CAGR per unit of maximum drawdown. | CAGR / absolute maximum drawdown. | Net performance metrics. | Whole run. | CAGR is annualized. | Undefined (NaN) when there is no drawdown. |
| Calmar Ratio | Parameter-sweep name for `Calmar`. | Calendar-time CAGR / absolute maximum drawdown. | Net performance metrics. | Whole run. | Calendar-time CAGR. | Same definition as `Calmar`. |
| Max Drawdown | Parameter-sweep name for `Maximum Drawdown`. | Minimum of `value / cumulative_max(value) - 1`. | Net daily portfolio values. | Worst date in the run. | None. | Same definition as `Maximum Drawdown`. |
| Average Realized Volatility | Realized volatility of the net portfolio return series. | Sample standard deviation of daily net returns × √252. | Net daily portfolio values. | One estimate over the full run. | 252 sessions/year. | A historical realized metric, not a forward-looking estimate. |
| Downside Volatility | Annualized downside deviation of net daily returns. | `sqrt(mean(min(return, 0) ** 2)) × √252`. | Net daily portfolio values. | Whole run's daily returns. | 252 sessions/year. | Uses zero as the minimum acceptable return. |
| Largest Loss Month | Worst monthly net return. | Lowest month-end value return; first month's return is measured from initial capital. | Net portfolio values. | Calendar months. | None. | Partial first and last months are included. |
| Negative Months | Number of months with negative net return. | Count of monthly returns below zero. | Net portfolio values. | Calendar months. | None. | Partial first and last months are included. |
| Positive Months | Number of months with positive net return. | Count of monthly returns above zero, with the first month's return measured from initial capital. | Net portfolio values and initial capital. | Calendar months. | None. | Partial first and last months are included; zero-return months are not counted. |
| Positive Month Share | Fraction of months with positive net return. | Count of positive monthly returns / number of monthly returns. | Net portfolio values. | Calendar months. | None. | Zero-return months are not positive. |
| Profit Factor | Gross profits divided by gross losses on closed VectorBT trades. | Sum of positive closed-trade PnL / absolute sum of negative closed-trade PnL. | Closed VectorBT trade records. | All closed trade records. | None. | Infinity if there are profits and no losses; NaN if neither exists. JSON encodes an infinite result as the string `"Infinity"` because JSON has no non-finite numeric literal. |
| Number of Trades | Number of closed records in VectorBT's trade accounting. | Length of `result.trades.closed.records_readable`. | Net VectorBT trade records. | Whole run. | None. | Not an order count, FIFO buy-lot count, or guaranteed count of position episodes. |
| Buy Orders | Number of executed buy order records. | Count order records with side `Buy`. | Net VectorBT order records. | Whole run. | None. | A partially filled broker order is represented according to VectorBT's order records. |
| Sell Orders | Number of executed sell order records. | Count order records with side `Sell`. | Net VectorBT order records. | Whole run. | None. | A sell can close or partially reduce FIFO lots. |
| Average Exposure | Average fraction of net portfolio value invested in long positions. | Per date, sum of held shares × last available close / net value; then arithmetic mean. | Net holdings, adjusted EUR closes, and net portfolio values. | All valuation dates, including dates with zero exposure. | None. | Stale closes are carried forward across missing dates. |
| Maximum Exposure | Maximum daily invested fraction of net portfolio value. | Maximum of the daily exposure series. | Same as `Average Exposure`. | Highest valuation date. | None. | May differ from target exposure and can drift with prices. |
| Cash Allocation | Average fraction not invested in reported long positions. | Arithmetic mean of `1 - daily exposure`. | Same as `Average Exposure`. | All valuation dates. | None. | Complements exposure under the long-only, unlevered portfolio accounting. |
| Average Cash Allocation | Rolling-report alias for `Cash Allocation`. | Same as `Cash Allocation`. | Same as `Average Exposure`. | All valuation dates in the window. | None. | Alias; no separate definition. |
| Time with Cash | Fraction of valuation dates with exposure below 100%. | Count of dates where exposure < 1 - 1e-9 / all valuation dates. | Same as `Average Exposure`. | All valuation dates. | None. | It counts any cash remainder, including one caused by fees or unused target allocation. |
| Average EUROPE Exposure | Average portfolio-value fraction in instruments classified as EUROPE. | Daily regional market value / net value, averaged across dates. | Held shares, carried-forward closes, and country metadata. | All valuation dates. | None. | Countries outside the configured Europe list fall into `OTHER`. |
| Maximum EUROPE Exposure | Maximum daily EUROPE fraction. | Maximum of daily EUROPE exposure. | Same as `Average EUROPE Exposure`. | Highest valuation date. | None. | Classification depends on available country metadata. |
| Average USA Exposure | Average portfolio-value fraction in United States instruments. | Daily regional market value / net value, averaged across dates. | Held shares, carried-forward closes, and country metadata. | All valuation dates. | None. | Classification depends on exact country metadata. |
| Maximum USA Exposure | Maximum daily USA fraction. | Maximum of daily USA exposure. | Same as `Average USA Exposure`. | Highest valuation date. | None. | Classification depends on exact country metadata. |
| Average OTHER Exposure | Average portfolio-value fraction in all other instruments. | Daily regional market value / net value, averaged across dates. | Held shares, carried-forward closes, and country metadata. | All valuation dates. | None. | Includes instruments without a recognized country. |
| Maximum OTHER Exposure | Maximum daily OTHER fraction. | Maximum of daily OTHER exposure. | Same as `Average OTHER Exposure`. | Highest valuation date. | None. | Classification depends on available country metadata. |
| Turnover | Full-run two-sided turnover. | Buy plus sell notional / mean daily net portfolio value. | Net executed orders and net portfolio values. | Whole run. | None. | Uses fill price and absolute filled quantity. |
| One-sided Turnover | Full-run purchase-only turnover. | Buy notional / mean daily net portfolio value. | Net executed buy orders and net portfolio values. | Whole run. | None. | Excludes sell notional. |
| Two-sided Turnover | Full-run purchase-plus-sale turnover. | (Buy notional + sell notional) / mean daily net portfolio value. | Net executed orders and net portfolio values. | Whole run. | None. | Alias underlying `Turnover`. |
| Annual Portfolio Turnover | Annualized two-sided full-run turnover. | Two-sided turnover / (observed valuation sessions / 252). | Net executed orders and net portfolio values. | Whole run. | 252 sessions/year. | Alias for `Annualized Two-sided Turnover`. |
| Annualized One-sided Turnover | Annualized purchase-only turnover. | One-sided turnover / (observed valuation sessions / 252). | Net executed orders and net portfolio values. | Whole run. | 252 sessions/year. | Uses observed valuation rows, not exchange-specific session counts. |
| Annualized Two-sided Turnover | Annualized purchase-plus-sale turnover. | Two-sided turnover / (observed valuation sessions / 252). | Net executed orders and net portfolio values. | Whole run. | 252 sessions/year. | Uses observed valuation rows, not exchange-specific session counts. |
| Annualized One-sided Turnover {year} | Annualized purchase-only turnover for a calendar year. | Buy notional in that order year / that year's mean daily portfolio value / (observed year sessions / 252). | Orders grouped by execution timestamp year and daily values. | Each calendar year represented in portfolio values. | 252 sessions/year. | Orders are assigned to their execution year. |
| Annualized Two-sided Turnover {year} | Annualized purchase-plus-sale turnover for a calendar year. | Buy plus sell notional in that order year / that year's mean daily portfolio value / (observed year sessions / 252). | Orders grouped by execution timestamp year and daily values. | Each calendar year represented in portfolio values. | 252 sessions/year. | Orders are assigned to their execution year. |
| Completed Trade Lots | Number of FIFO buy lots fully exhausted by sales. | Count each buy lot once, only when all shares from that buy are sold. | Net order records, FIFO matched by instrument and execution timestamp. | Whole run. | None. | A partial sale is not a completed lot; open lots are excluded. |
| Average Holding Period (sessions) | Arithmetic mean duration of completed FIFO buy lots. | Mean of buy-session to final-lot-sale-session differences. | Net orders and instrument session dates; mapped exchanges use their exchange calendar. | One duration per completed buy lot. | None. | Entry session is not counted as elapsed; unmatched open lots are excluded. |
| Median Holding Period (sessions) | Median duration of completed FIFO buy lots. | Median of the same completed-lot durations as the average. | Same as `Average Holding Period (sessions)`. | One duration per completed buy lot. | None. | Partial sale portions do not create extra lots or observations. |
| Weighted Average Holding Period (sessions) | Completed-lot holding duration weighted by original lot size. | Sum(duration × original buy quantity) / sum(original buy quantity) for completed lots. | Same as `Average Holding Period (sessions)`. | Completed buy lots, weighted by shares. | None. | Open lots are excluded; lot duration ends on its final sale. |
| Completed Positions | Number of instrument holding episodes fully closed. | Count each transition from flat to nonzero inventory and back to flat. | Net order records grouped by instrument. | Whole run, with repeated episodes per instrument counted separately. | None. | Distinct from VectorBT trade records and buy lots. |
| Average Position Duration (sessions) | Mean duration of fully closed instrument holding episodes. | Mean from first buy in an episode to the sale that returns inventory to zero. | Net orders and instrument session dates. | One duration per completed position episode. | None. | Entry session is not counted as elapsed. |
| Open Lots | Number of FIFO buy lots with shares remaining at the final date. | Count remaining positive-quantity buy lots by instrument. | Net orders matched FIFO. | Final backtest state. | None. | Open inventory is not counted as completed lots. |
| Open Positions | Number of instruments with nonzero FIFO inventory at the final date. | Count instruments whose remaining lots are nonempty. | Net orders matched FIFO. | Final backtest state. | None. | Open positions are not counted as completed positions. |
| Average Open Position Duration (sessions) | Mean age of positions still open at the final date. | Mean from first buy in each open instrument episode to its last session in the backtest. | Net orders, instrument calendars where mapped, and close dates otherwise. | One duration per open instrument. | None. | Not a mark-to-market holding-period estimate for each open lot. |
| Proportional Fees Costs | Total proportional commissions recorded on executed orders. | Sum(abs(filled quantity) × fill price × configured proportional fee rate). | Net order records and broker fee configuration. | Whole run. | None. | Fees already reduce net portfolio value. |
| Fixed Fees Costs | Total fixed commissions recorded on executed order records. | Recorded order fees minus proportional fees. | Net order records and broker fee configuration. | Whole run; fixed component per executed order record. | None. | Fees already reduce net portfolio value. |
| Transaction Fees | All recorded commissions. | Sum of `Fees` across net order records. | Net VectorBT order records. | Whole run. | None. | Equals proportional plus fixed fees when the recorded fee schedule is consistent. |
| Slippage Costs | Difference between fill prices and next-open reference prices. | Sum(abs(filled quantity) × abs(fill price - reference execution price)). | Net order records and prepared execution-price panel. | Whole run, buys and sells. | None. | Slippage is already charged in net simulation. |
| Transaction Costs | Total reported commissions plus slippage. | Transaction fees + slippage costs. | Same as component metrics. | Whole run. | None. | Report-only; never deducted a second time from net returns. |
| Annual Transaction Costs | Average annual cost amount over the elapsed calendar years. | Whole-run transaction costs / elapsed calendar years. | Net order records and execution prices. | Whole run. | Calendar-year denominator (days / 365.25). | The annual total is an average, not a per-calendar-year sum. |
| Annual Transaction Costs {year} | Commissions plus slippage for orders executed in a calendar year. | Fees and slippage for orders whose execution timestamp has that year. | Net order records and execution prices. | By order execution year. | None. | The final year can be partial. |
| Cost Share of Gross Profit | Transaction costs as a share of gross portfolio profit. | Transaction costs / (gross final value - initial capital). | Net cost records, gross simulation, and broker configuration. | Whole run. | None. | NaN when gross profit is non-positive; can exceed 100%. |
| Average Position Weight | Mean actual weight across all active instrument-date pairs. | Mean of position market value / net portfolio value where weight > 1e-10. | Net held shares, carried-forward adjusted closes, net portfolio values. | Pooled across active positions and dates; not the mean of daily averages. | None. | Actual weights can drift above the rebalance target cap. |
| Median Position Weight | Median actual weight across active instrument-date pairs. | Median of the same pooled active weights as the average. | Same as `Average Position Weight`. | Pooled across active positions and dates. | None. | Actual weights can drift above the rebalance target cap. |
| Largest Position Weight | Largest actual single-instrument weight observed. | Maximum active position market value / net portfolio value. | Same as `Average Position Weight`. | Maximum over positions and dates. | None. | This is actual exposure, not the target weight limit. |
| Average Position Volatility | Mean annualized volatility of held constituents. | Mean constituent value over active position-dates; constituent volatility is the equal blend of 20- and 60-session sample standard deviations × √252. | Instrument adjusted closes and actual holdings. | Pooled across active instrument-date pairs with valid volatility. | 252 sessions/year. | Constituents are not covariance-adjusted. |
| Weight-Weighted Average Constituent Volatility | Mean across dates of the invested-weighted constituent volatility. | Per date, sum(position weight × constituent volatility) / sum(weights with valid volatility); then mean. | Actual weights and the 20/60-session constituent volatility series. | Daily weighted constituent estimate, averaged over dates where defined. | 252 sessions/year within constituent volatility. | Not portfolio volatility; ignores correlations. Replaces the misleading old label `Weighted Average Portfolio Volatility`. |
| Average Covariance-Based Portfolio Volatility | Mean point-in-time annualized covariance estimate for actual holdings. | For each date, `sqrt(wᵀΣw) × √252`, where Σ is sample covariance of the last at most 60 jointly observed daily returns for instruments held that date; minimum 20 complete observations. Cash has zero return and is implicit. | Adjusted closes, actual close-marked weights, and exchange calendars for mapped instruments. | Daily estimates averaged over dates; all-cash dates contribute zero. | 252 sessions/year. | Uses complete same-date observations only; insufficient history yields NaN. Unmapped calendars and asynchronous market closes can reduce observations. Estimate is close-of-date, not intraday. |

## Rolling-window aggregation

The rolling robustness output is generated from the same per-window result
rows as its detail CSV, detail JSON, and HTML report. `Sharpe Ratio`, `Sortino
Ratio`, `Calmar Ratio`, and `Average Cash Allocation` are report aliases for
the identically defined metrics shown above; they do not recalculate returns.
`Positive Months` uses the same monthly return series as `Negative Months` and
`Positive Month Share`. Cost-to-capital metrics divide the named absolute cost
by that window's starting capital and remain decimal-valued.

Each nominal three-year window is `[start_date, end_date)`. A dataset-level
window is complete if its nominal end is no later than the calendar day after
the latest date with any local close observation. Incomplete windows are run
only through the latest observed date and are retained in detail, but excluded
from summary statistics, charts, and aggregate pairwise comparisons. Per-window
position, exposure, cash, volatility, trade, and cost metrics retain their
definitions above. Summary min/max/mean/median/sample standard deviation treat
each complete window as one observation; they do not pool daily observations
across overlapping windows. The sample standard deviation uses `ddof=1`.
Positive/negative window counts use `Net Total Return`; a return of exactly
zero is neither positive nor negative. Window overlap makes these observations
dependent, so dispersion is descriptive rather than an independent-sample
estimate.

Pairwise differences are reported as variant B minus reference variant A.
Higher CAGR, Sharpe, and maximum drawdown (a less-negative drawdown) are
considered better; lower transaction costs are better. The share of windows in
which A is better is calculated over complete windows with both values present.
The rolling report uses existing configured strategy, cost, and execution
definitions; it does not modify signal, sizing, rebalance, or order behavior.
Each detail row's `Null Metric Reasons` field is a JSON object keyed by null
metric or annual field. It distinguishes absent calendar-year observations,
known undefined-metric conditions, and variant failures; the null metric itself
remains null rather than being replaced by zero.
| Weight Concentration HHI | Average Herfindahl-Hirschman concentration of invested holdings. | Per invested date, sum((position weight / total invested exposure)²); mean over invested dates. | Actual close-marked position weights. | Daily invested-portfolio HHI, excluding zero-exposure dates. | None. | Cash is excluded from the normalization; HHI is not normalized to [0, 1] by asset count. |
| Time at or Above Position Cap | Fraction of all portfolio valuation dates with a position at or above the configured cap. | Count dates where maximum actual position weight ≥ `max_position_weight - 1e-6` / all valuation dates. | Actual close-marked weights and configured target cap. | All valuation dates, including dates with no holdings. | None. | Tests actual weights; the 20% limit is only a target-weight cap at scheduled rebalances, not an intraperiod hard limit. |

## Cost and output conventions

Net results already include proportional fees, fixed order fees, and slippage.
Gross results rerun the same prepared target weights with all three cost
parameters set to zero. Reported costs are diagnostic components of the net
simulation and are not subtracted again. Open positions remain in final
portfolio value at the final available valuation price; they do not count as
completed lots or completed positions.

CSV, JSON, and HTML metric values come from the same flattened result rows.
Missing values are blank in CSV/HTML and `null` in JSON. Positive and negative
infinity are written as `inf`/`-inf` in CSV/HTML and `"Infinity"`/`"-Infinity"`
strings in JSON, preserving their mathematical meaning within JSON's standard
value types.
`config.json` remains the reproducibility/configuration archive and is not a
second metrics export. The configured 20% maximum position weight applies only
to target weights at regular rebalances: actual weights and `Largest Position
Weight` may exceed it between rebalances without forced sales.

The separate `parameter_sweep` API uses `pytrading.stats.stats`; its headline
names (`Sharpe Ratio`, `Calmar Ratio`, `CAGR`, `Max Drawdown`, and `Profit
Factor`) now share the definitions above rather than VectorBT's different
observation-frequency annualization.
