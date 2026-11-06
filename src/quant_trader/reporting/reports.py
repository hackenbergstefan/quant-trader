from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_trader import __version__
from quant_trader.backtesting.engine import BacktestResult
from quant_trader.config import AppConfig
from quant_trader.reporting.charts import write_charts
from quant_trader.reporting.notebooks import write_report_notebook


def configuration_hash(config: AppConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def data_hash(bars: pd.DataFrame) -> str:
    stable = bars.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(stable, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_report(
    result: BacktestResult,
    config: AppConfig,
    bars: pd.DataFrame,
    output_dir: str | Path,
    *,
    data_source: str,
    additional_metadata: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_digest = configuration_hash(config)
    source_values = (
        sorted(bars["data_source"].dropna().astype(str).unique())
        if "data_source" in bars
        else [data_source]
    )
    metadata = {
        **result.metadata,
        "strategy_version": __version__,
        "configuration_hash": config_digest,
        "data_hash": data_hash(bars),
        "data_source": data_source,
        "source_labels": source_values,
        "data_coverage": {
            "start": pd.to_datetime(bars["timestamp"]).min().date().isoformat(),
            "end": pd.to_datetime(bars["timestamp"]).max().date().isoformat(),
            "instrument_count": int(bars["instrument_id"].nunique()),
            "bar_count": int(len(bars)),
        },
        "tax_treatment": "Tax-neutral; taxes are not modeled.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **(additional_metadata or {}),
    }
    report = {
        "metadata": metadata,
        "configuration": config.model_dump(mode="json"),
        "metrics": result.metrics,
    }
    json_path = root / "report.json"
    json_path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# Backtest report: {metadata['strategy']}",
        "",
        f"- Period: {metadata['start_date']} to {metadata['end_date']}",
        f"- Data source: {data_source}",
        f"- Data coverage: {metadata['data_coverage']['start']} to {metadata['data_coverage']['end']}",
        f"- Instruments: {metadata['data_coverage']['instrument_count']} ({metadata['data_coverage']['bar_count']} bars)",
        f"- Strategy version: {metadata['strategy_version']}",
        f"- Configuration SHA-256: `{config_digest}`",
        f"- Input data SHA-256: `{metadata['data_hash']}`",
        f"- Execution: {metadata.get('stop_model', 'daily close signals, next observed open execution')}",
        f"- Tax treatment: {metadata['tax_treatment']}",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric, value in result.metrics.items():
        display = (
            "N/A"
            if value is None
            else f"{value:,.4f}"
            if isinstance(value, (float, int))
            else str(value)
        )
        lines.append(f"| {metric.replace('_', ' ').title()} | {display} |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result.equity_curve.to_csv(root / "equity_curve.csv", index=False)
    result.transactions.to_csv(root / "transactions.csv", index=False)
    result.trades.to_csv(root / "trades.csv", index=False)
    write_charts(result.equity_curve, root, str(metadata["strategy"]))
    write_report_notebook(report, root)
    return json_path
