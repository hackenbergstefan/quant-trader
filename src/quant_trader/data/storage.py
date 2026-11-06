from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


class MarketDataStore:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.database))
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                instrument_id VARCHAR, symbol VARCHAR, exchange VARCHAR, currency VARCHAR,
                calendar VARCHAR, fx_to_eur DOUBLE, timestamp TIMESTAMP, open DOUBLE,
                high DOUBLE, low DOUBLE, close DOUBLE, adjusted_close DOUBLE, volume DOUBLE,
                data_source VARCHAR, retrieved_at VARCHAR,
                PRIMARY KEY (instrument_id, timestamp)
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS raw_prices (
                instrument_id VARCHAR, timestamp TIMESTAMP, data_source VARCHAR,
                retrieved_at VARCHAR, payload VARCHAR,
                PRIMARY KEY (instrument_id, timestamp, retrieved_at)
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                instrument_id VARCHAR, ex_date DATE, action_type VARCHAR, value DOUBLE,
                currency VARCHAR, PRIMARY KEY (instrument_id, ex_date, action_type)
            )
        """)

    def save_raw(
        self, raw: pd.DataFrame, instrument_id: str, data_source: str, retrieved_at: str
    ) -> None:
        if raw.empty:
            return
        raw_copy = raw.copy()
        if isinstance(raw_copy.index, pd.DatetimeIndex):
            raw_copy = raw_copy.reset_index(names="timestamp")
        raw_copy.columns = [
            str(column).strip().lower().replace(" ", "_") for column in raw_copy.columns
        ]
        if "timestamp" not in raw_copy:
            raw_copy = raw_copy.rename(columns={raw_copy.columns[0]: "timestamp"})
        raw_rows = []
        for _, row in raw_copy.iterrows():
            date_value = pd.Timestamp(row["timestamp"])
            if date_value.tzinfo is not None:
                date_value = date_value.tz_localize(None)
            payload = {
                str(key): (
                    None if pd.isna(value) else value.item() if hasattr(value, "item") else value
                )
                for key, value in row.drop(labels=["timestamp"]).items()
            }
            raw_rows.append(
                (
                    instrument_id,
                    date_value.normalize(),
                    data_source,
                    retrieved_at,
                    json.dumps(payload, default=str),
                )
            )
        self.connection.executemany(
            "INSERT OR REPLACE INTO raw_prices VALUES (?, ?, ?, ?, ?)", raw_rows
        )

    def save(
        self,
        raw: pd.DataFrame,
        normalized: pd.DataFrame,
        actions: pd.DataFrame | None = None,
        *,
        raw_already_saved: bool = False,
    ) -> None:
        if normalized.empty:
            return
        retrieved_at = str(normalized.iloc[0]["retrieved_at"])
        source = str(normalized.iloc[0]["data_source"])
        if not raw_already_saved:
            self.save_raw(raw, str(normalized.iloc[0]["instrument_id"]), source, retrieved_at)
        self.connection.register("normalized_frame", normalized)
        self.connection.execute("""
            INSERT OR REPLACE INTO prices SELECT * FROM normalized_frame
        """)
        self.connection.unregister("normalized_frame")
        if actions is not None and not actions.empty:
            self.connection.register("actions_frame", actions)
            self.connection.execute(
                "INSERT OR REPLACE INTO corporate_actions SELECT * FROM actions_frame"
            )
            self.connection.unregister("actions_frame")

    def load_prices(self, instrument_ids: list[str] | None = None) -> pd.DataFrame:
        if instrument_ids:
            placeholders = ",".join("?" for _ in instrument_ids)
            query = f"SELECT * FROM prices WHERE instrument_id IN ({placeholders}) ORDER BY timestamp, instrument_id"
            return self.connection.execute(query, instrument_ids).fetchdf()
        return self.connection.execute(
            "SELECT * FROM prices ORDER BY timestamp, instrument_id"
        ).fetchdf()

    def load_actions(self, instrument_ids: list[str] | None = None) -> pd.DataFrame:
        if instrument_ids:
            placeholders = ",".join("?" for _ in instrument_ids)
            query = f"SELECT * FROM corporate_actions WHERE instrument_id IN ({placeholders}) ORDER BY ex_date"
            return self.connection.execute(query, instrument_ids).fetchdf()
        return self.connection.execute("SELECT * FROM corporate_actions ORDER BY ex_date").fetchdf()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> MarketDataStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
