from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_trader.config import ExecutionConfig, RiskConfig
from quant_trader.domain.models import ActionType, CorporateAction
from quant_trader.domain.portfolio import Portfolio, Position
from quant_trader.portfolio.risk_manager import RiskManager


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    transactions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int | None]
    metadata: dict[str, object]


class BacktestEngine:
    """Close-time signals fill at the instrument's next observed open."""

    def __init__(
        self,
        risk: RiskConfig,
        execution: ExecutionConfig,
        *,
        stop_atr_multiple: float = 2.0,
        strategy_name: str = "adaptive_momentum",
        max_positions: int = 5,
    ) -> None:
        self.risk = risk
        self.execution = execution
        self.stop_atr_multiple = stop_atr_multiple
        self.strategy_name = strategy_name
        self.max_positions = max_positions

    def run(
        self,
        bars: pd.DataFrame,
        signals: pd.DataFrame,
        actions: list[CorporateAction] | None = None,
    ) -> BacktestResult:
        if bars.empty:
            raise ValueError("Cannot backtest without price bars")
        frame = bars.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
        frame = frame.sort_values(["timestamp", "instrument_id"], kind="stable")
        signals_frame = signals.copy()
        signals_frame["timestamp"] = pd.to_datetime(signals_frame["timestamp"]).dt.normalize()

        rows_by_day = {
            stamp: group.set_index("instrument_id")
            for stamp, group in frame.groupby("timestamp", sort=True)
        }
        next_session: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
        for instrument_id, group in frame.groupby("instrument_id", sort=False):
            timestamps = list(group["timestamp"])
            next_session.update(
                {(instrument_id, left): right for left, right in zip(timestamps, timestamps[1:])}
            )

        pending: dict[pd.Timestamp, dict[str, pd.Series]] = defaultdict(dict)
        for _, signal in signals_frame.sort_values("timestamp").iterrows():
            execution_day = next_session.get((signal["instrument_id"], signal["timestamp"]))
            if execution_day is not None:
                pending[execution_day][signal["instrument_id"]] = signal

        actions_by_day: dict[date, list[CorporateAction]] = defaultdict(list)
        for action in actions or []:
            actions_by_day[action.ex_date].append(action)
        all_days = sorted(set(rows_by_day) | {pd.Timestamp(day) for day in actions_by_day})

        portfolio = Portfolio(self.risk.starting_cash_eur, {})
        risk_manager = RiskManager(
            self.risk, self.stop_atr_multiple, self.execution.total_cost_rate
        )
        transactions: list[dict[str, object]] = []
        closed_trades: list[dict[str, object]] = []
        equity_rows: list[dict[str, object]] = []
        last_prices: dict[str, float] = {}
        fx_rates: dict[str, float] = {}
        blocked_after_stop: set[str] = set()

        for day in all_days:
            current_rows = rows_by_day.get(day)
            if current_rows is None:
                current_rows = pd.DataFrame()
            current_ids = set(current_rows.index) if not current_rows.empty else set()
            for instrument_id, row in current_rows.iterrows() if not current_rows.empty else []:
                fx_rates[instrument_id] = float(row.get("fx_to_eur", 1.0))

            for action in actions_by_day.get(day.date(), []):
                position = portfolio.positions.get(action.instrument_id)
                if position is None:
                    continue
                if action.action_type == ActionType.SPLIT:
                    if action.value <= 0:
                        raise ValueError(
                            f"Invalid split ratio {action.value} for {action.instrument_id}"
                        )
                    theoretical_quantity = position.quantity * action.value
                    position.quantity = math.floor(theoretical_quantity + 1e-12)
                    position.entry_quantity *= action.value
                    position.average_entry_price /= action.value
                    position.stop_price /= action.value
                    if action.instrument_id in last_prices:
                        last_prices[action.instrument_id] /= action.value
                    fraction = theoretical_quantity - position.quantity
                    if fraction > 1e-12 and action.instrument_id in current_ids:
                        split_row = current_rows.loc[action.instrument_id]
                        split_price = float(split_row["open"])
                        split_fx = float(split_row.get("fx_to_eur", 1.0))
                        cash_in_lieu = fraction * split_price * split_fx
                        cash_basis = fraction * position.average_entry_price * split_fx
                        portfolio.cash_eur += cash_in_lieu
                        position.realized_pnl_eur += cash_in_lieu - cash_basis
                        self._record_transaction(
                            transactions,
                            day,
                            action.instrument_id,
                            "cash_in_lieu",
                            fraction,
                            split_price,
                            split_fx,
                            0.0,
                            "fractional_split_share",
                            portfolio,
                        )
                elif action.action_type == ActionType.DIVIDEND:
                    fx = fx_rates.get(action.instrument_id, position.fx_to_eur)
                    cash_dividend = position.quantity * action.value * fx
                    portfolio.cash_eur += cash_dividend
                    position.dividends_eur += cash_dividend
                    position.stop_price = max(0.0, position.stop_price - action.value)

            for instrument_id, row in current_rows.iterrows() if not current_rows.empty else []:
                last_prices[instrument_id] = float(row["open"])
                fx_rates[instrument_id] = float(row.get("fx_to_eur", 1.0))

            due = pending.get(day, {})
            target_rows = {
                instrument_id: signal
                for instrument_id, signal in due.items()
                if instrument_id in current_ids
            }

            def open_price(instrument_id: str) -> float:
                return float(current_rows.loc[instrument_id, "open"])

            def fx_rate(instrument_id: str) -> float:
                return float(current_rows.loc[instrument_id].get("fx_to_eur", 1.0))

            def mark_to_open() -> tuple[float, float]:
                invested = sum(
                    position.quantity
                    * last_prices.get(instrument_id, position.average_entry_price)
                    * fx_rates.get(instrument_id, position.fx_to_eur)
                    for instrument_id, position in portfolio.positions.items()
                )
                return portfolio.cash_eur + invested, invested

            to_exit = [
                key for key, signal in target_rows.items() if float(signal["target_weight"]) <= 0
            ]
            for instrument_id in sorted(to_exit):
                position = portfolio.positions.get(instrument_id)
                if position:
                    self._sell_all(
                        portfolio,
                        position,
                        day,
                        open_price(instrument_id),
                        fx_rate(instrument_id),
                        "signal_exit",
                        transactions,
                        closed_trades,
                    )

            # Reductions are processed before increases so released cash is available at the same open.
            ranked_instruments = sorted(
                target_rows,
                key=lambda instrument_id: (
                    -float(target_rows[instrument_id].get("momentum_score", 0.0)),
                    instrument_id,
                ),
            )
            for instrument_id in ranked_instruments:
                signal = target_rows[instrument_id]
                position = portfolio.positions.get(instrument_id)
                if position is None or float(signal["target_weight"]) <= 0:
                    continue
                price, fx = open_price(instrument_id), fx_rate(instrument_id)
                equity, invested = mark_to_open()
                desired = self._desired_shares(
                    position=position,
                    signal=signal,
                    price=price,
                    fx=fx,
                    equity=equity,
                    invested=invested,
                    cash=portfolio.cash_eur,
                    risk_manager=risk_manager,
                )
                current_value = position.quantity * price * fx
                target_value = desired * price * fx
                drift = abs(target_value - current_value) / max(current_value, target_value, 1.0)
                if desired < position.quantity and drift >= self.risk.rebalance_threshold:
                    self._sell(
                        portfolio,
                        position,
                        position.quantity - desired,
                        day,
                        price,
                        fx,
                        "rebalance",
                        transactions,
                        closed_trades,
                    )

            for instrument_id in ranked_instruments:
                signal = target_rows[instrument_id]
                if float(signal["target_weight"]) <= 0:
                    continue
                price, fx = open_price(instrument_id), fx_rate(instrument_id)
                position = portfolio.positions.get(instrument_id)
                equity, invested = mark_to_open()
                if position is None:
                    if len(portfolio.positions) >= self.max_positions:
                        continue
                    atr = signal.get("atr")
                    if pd.isna(atr) or float(atr) <= 0:
                        continue
                    shares = risk_manager.shares_for_entry(
                        equity_eur=equity,
                        cash_eur=portfolio.cash_eur,
                        entry_price=price,
                        fx_to_eur=fx,
                        atr=float(atr),
                        target_weight=float(signal["target_weight"]),
                        invested_eur=invested,
                    )
                    if shares > 0:
                        self._buy(
                            portfolio,
                            instrument_id,
                            shares,
                            day,
                            price,
                            fx,
                            float(atr),
                            equity * self.risk.risk_budget_per_position,
                            "signal_entry",
                            transactions,
                        )
                else:
                    desired = self._desired_shares(
                        position=position,
                        signal=signal,
                        price=price,
                        fx=fx,
                        equity=equity,
                        invested=invested,
                        cash=portfolio.cash_eur,
                        risk_manager=risk_manager,
                    )
                    current_value = position.quantity * price * fx
                    target_value = desired * price * fx
                    drift = abs(target_value - current_value) / max(
                        current_value, target_value, 1.0
                    )
                    if desired > position.quantity and drift >= self.risk.rebalance_threshold:
                        self._buy(
                            portfolio,
                            instrument_id,
                            desired - position.quantity,
                            day,
                            price,
                            fx,
                            float(signal["atr"]),
                            position.risk_budget_eur,
                            "rebalance",
                            transactions,
                        )

            # A stop triggered by an opening gap fills at that open; an intraday touch fills at the stop.
            for instrument_id in sorted(current_ids):
                position = portfolio.positions.get(instrument_id)
                if position is None:
                    continue
                row = current_rows.loc[instrument_id]
                if float(row["open"]) <= position.stop_price:
                    stop_fill, reason = float(row["open"]), "stop_gap"
                elif float(row["low"]) <= position.stop_price:
                    stop_fill, reason = position.stop_price, "stop_trigger"
                else:
                    continue
                self._sell_all(
                    portfolio,
                    position,
                    day,
                    stop_fill,
                    fx_rate(instrument_id),
                    reason,
                    transactions,
                    closed_trades,
                )
                blocked_after_stop.add(instrument_id)

            for instrument_id, row in current_rows.iterrows() if not current_rows.empty else []:
                last_prices[instrument_id] = float(row["close"])
                fx_rates[instrument_id] = float(row.get("fx_to_eur", 1.0))

            ending_instruments = sorted(
                instrument_id
                for instrument_id in current_ids
                if next_session.get((instrument_id, day)) is None
            )
            for instrument_id in ending_instruments:
                position = portfolio.positions.get(instrument_id)
                if position is not None:
                    self._sell_all(
                        portfolio,
                        position,
                        day,
                        last_prices[instrument_id],
                        fx_rates[instrument_id],
                        "terminal_liquidation",
                        transactions,
                        closed_trades,
                    )

            if day in rows_by_day:
                close_equity = portfolio.cash_eur + sum(
                    position.quantity
                    * last_prices.get(instrument_id, position.average_entry_price)
                    * fx_rates.get(instrument_id, position.fx_to_eur)
                    for instrument_id, position in portfolio.positions.items()
                )
                exposure = close_equity - portfolio.cash_eur
                equity_rows.append(
                    {
                        "timestamp": day,
                        "equity": close_equity,
                        "cash": portfolio.cash_eur,
                        "exposure_eur": exposure,
                        "exposure_ratio": exposure / close_equity if close_equity > 0 else 0.0,
                        "transaction_costs_eur": portfolio.transaction_costs_eur,
                    }
                )

            if day in rows_by_day:
                day_signals = signals_frame.loc[signals_frame["timestamp"].eq(day)]
                for _, signal in day_signals.iterrows():
                    instrument_id = signal["instrument_id"]
                    selected = bool(signal["selected"])
                    weight = float(signal["target_weight"])
                    if instrument_id in blocked_after_stop:
                        if selected:
                            weight = 0.0
                        else:
                            blocked_after_stop.remove(instrument_id)
                            weight = 0.0
                    signal = signal.copy()
                    signal["target_weight"] = weight
                    execution_day = next_session.get((instrument_id, day))
                    if execution_day is not None:
                        pending[execution_day][instrument_id] = signal

        equity_curve = pd.DataFrame(equity_rows)
        if equity_curve.empty:
            raise ValueError("No daily valuations were produced")
        equity_curve["daily_return"] = equity_curve["equity"].pct_change().fillna(0.0)
        transactions_frame = pd.DataFrame(transactions)
        trades_frame = pd.DataFrame(closed_trades)
        metrics = calculate_metrics(equity_curve, transactions_frame, trades_frame)
        metadata = {
            "strategy": self.strategy_name,
            "start_date": equity_curve["timestamp"].min().date().isoformat(),
            "end_date": equity_curve["timestamp"].max().date().isoformat(),
            "starting_cash_eur": self.risk.starting_cash_eur,
            "execution_cost_bps": self.execution.total_cost_rate * 10_000,
            "execution_cost_components_bps": {
                "commission": self.execution.commission_bps,
                "exchange_fee": self.execution.exchange_fee_bps,
                "spread": self.execution.spread_bps,
                "slippage": self.execution.slippage_bps,
            },
            "stop_model": "2*ATR stop; gap fills at open, intraday touch at stop, plus configured costs",
            "terminal_position_model": "Positions are liquidated at each instrument's final available close, with configured costs.",
        }
        return BacktestResult(equity_curve, transactions_frame, trades_frame, metrics, metadata)

    def _desired_shares(
        self,
        *,
        position: Position,
        signal: pd.Series,
        price: float,
        fx: float,
        equity: float,
        invested: float,
        cash: float,
        risk_manager: RiskManager,
    ) -> int:
        atr = float(signal["atr"]) if not pd.isna(signal.get("atr")) else 0.0
        if atr <= 0 or price <= 0 or fx <= 0:
            return 0
        stop_distance = max(
            position.average_entry_price - position.stop_price, atr * self.stop_atr_multiple
        )
        risk_shares = math.floor(position.risk_budget_eur / (stop_distance * fx))
        exposure_room = max(0.0, equity * self.risk.max_portfolio_exposure - invested)
        target_notional = min(
            float(signal["target_weight"]) * equity,
            self.risk.max_position_value_eur,
            exposure_room,
        )
        desired = math.floor(target_notional / (price * fx))
        desired = min(desired, risk_shares)
        if desired > position.quantity:
            max_additional = math.floor(cash / (price * fx * (1 + risk_manager.cost_rate)))
            desired = min(desired, position.quantity + max_additional)
        return max(0, desired)

    def _buy(
        self,
        portfolio: Portfolio,
        instrument_id: str,
        quantity: int,
        day: pd.Timestamp,
        price: float,
        fx: float,
        atr: float,
        risk_budget: float,
        reason: str,
        transactions: list[dict[str, object]],
    ) -> None:
        if quantity <= 0:
            return
        notional = quantity * price * fx
        costs = notional * self.execution.total_cost_rate
        while quantity > 0 and notional + costs > portfolio.cash_eur + 1e-8:
            quantity -= 1
            notional = quantity * price * fx
            costs = notional * self.execution.total_cost_rate
        if quantity <= 0:
            return
        portfolio.cash_eur -= notional + costs
        portfolio.transaction_costs_eur += costs
        position = portfolio.positions.get(instrument_id)
        if position is None:
            position = Position(
                instrument_id,
                quantity,
                price,
                day.date(),
                price - atr * self.stop_atr_multiple,
                quantity,
                costs,
                0.0,
                0.0,
                risk_budget,
                fx,
            )
            portfolio.positions[instrument_id] = position
        else:
            total_quantity = position.quantity + quantity
            position.average_entry_price = (
                position.average_entry_price * position.quantity + price * quantity
            ) / total_quantity
            position.quantity = total_quantity
            position.entry_quantity += quantity
            position.entry_costs_eur += costs
        self._record_transaction(
            transactions, day, instrument_id, "buy", quantity, price, fx, costs, reason, portfolio
        )

    def _sell_all(
        self,
        portfolio: Portfolio,
        position: Position,
        day: pd.Timestamp,
        price: float,
        fx: float,
        reason: str,
        transactions: list[dict[str, object]],
        closed_trades: list[dict[str, object]],
    ) -> None:
        self._sell(
            portfolio,
            position,
            position.quantity,
            day,
            price,
            fx,
            reason,
            transactions,
            closed_trades,
        )

    def _sell(
        self,
        portfolio: Portfolio,
        position: Position,
        quantity: float,
        day: pd.Timestamp,
        price: float,
        fx: float,
        reason: str,
        transactions: list[dict[str, object]],
        closed_trades: list[dict[str, object]],
    ) -> None:
        quantity = min(quantity, position.quantity)
        if quantity <= 0:
            return
        starting_quantity = position.quantity
        notional = quantity * price * fx
        costs = notional * self.execution.total_cost_rate
        allocated_entry_cost = position.entry_costs_eur * quantity / starting_quantity
        allocated_dividend = position.dividends_eur * quantity / starting_quantity
        realized = (
            quantity * (price - position.average_entry_price) * fx
            - costs
            - allocated_entry_cost
            + allocated_dividend
        )
        portfolio.cash_eur += notional - costs
        portfolio.transaction_costs_eur += costs
        position.realized_pnl_eur += realized
        position.entry_costs_eur -= allocated_entry_cost
        position.dividends_eur -= allocated_dividend
        position.quantity -= quantity
        self._record_transaction(
            transactions,
            day,
            position.instrument_id,
            "sell",
            quantity,
            price,
            fx,
            costs,
            reason,
            portfolio,
        )
        if position.quantity == 0:
            closed_trades.append(
                {
                    "instrument_id": position.instrument_id,
                    "entry_date": position.entry_date,
                    "exit_date": day.date(),
                    "quantity": position.entry_quantity,
                    "entry_price": position.average_entry_price,
                    "exit_price": price,
                    "pnl_eur": position.realized_pnl_eur,
                    "holding_days": (day.date() - position.entry_date).days,
                    "exit_reason": reason,
                }
            )
            del portfolio.positions[position.instrument_id]

    @staticmethod
    def _record_transaction(
        transactions: list[dict[str, object]],
        day: pd.Timestamp,
        instrument_id: str,
        side: str,
        quantity: float,
        price: float,
        fx: float,
        costs: float,
        reason: str,
        portfolio: Portfolio,
    ) -> None:
        transactions.append(
            {
                "timestamp": day.date(),
                "instrument_id": instrument_id,
                "side": side,
                "quantity": quantity,
                "price": price,
                "notional_eur": quantity * price * fx,
                "costs_eur": costs,
                "reason": reason,
                "cash_after_eur": portfolio.cash_eur,
            }
        )


def calculate_metrics(
    equity: pd.DataFrame,
    transactions: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float | None = None,
) -> dict[str, float | int | None]:
    values = equity["equity"].astype(float)
    returns = values.pct_change().dropna()
    years = max(len(returns) / 252, 1 / 252)
    initial_value = initial_capital if initial_capital is not None else float(values.iloc[0])
    total_return = float(values.iloc[-1] / initial_value - 1)
    peak = values.cummax()
    drawdown = values.div(peak).sub(1)
    volatility = float(returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_deviation = float(downside.std(ddof=1) * math.sqrt(252)) if len(downside) > 1 else 0.0
    daily_vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if trades.empty:
        wins, losses = pd.Series(dtype=float), pd.Series(dtype=float)
    else:
        pnl = trades["pnl_eur"].astype(float)
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    total_costs = float(transactions["costs_eur"].sum()) if not transactions.empty else 0.0
    average_equity = float(values.mean())
    turnover = float(transactions["notional_eur"].sum()) if not transactions.empty else 0.0
    return {
        "total_return_net": total_return,
        "total_return_gross_estimate": float(
            (values.iloc[-1] + equity["transaction_costs_eur"].iloc[-1]) / initial_value - 1
        ),
        "cagr_net": float((values.iloc[-1] / initial_value) ** (1 / years) - 1)
        if values.iloc[-1] > 0
        else -1.0,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": volatility,
        "sharpe_ratio": float(returns.mean() / daily_vol * math.sqrt(252))
        if daily_vol > 0
        else 0.0,
        "sortino_ratio": float(returns.mean() * 252 / downside_deviation)
        if downside_deviation > 0
        else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "win_rate": float(len(wins) / len(trades)) if len(trades) else 0.0,
        "average_win_eur": float(wins.mean()) if len(wins) else 0.0,
        "average_loss_eur": float(losses.mean()) if len(losses) else 0.0,
        "number_of_trades": int(len(trades)),
        "average_holding_days": float(trades["holding_days"].mean()) if len(trades) else 0.0,
        "annual_turnover": turnover / average_equity / years if average_equity else 0.0,
        "average_capital_exposure": float(equity["exposure_ratio"].mean()),
        "transaction_costs_eur": total_costs,
    }
