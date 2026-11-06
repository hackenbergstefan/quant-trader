from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from quant_trader.data.corporate_actions import actions_from_frame
from quant_trader.data.instruments import InstrumentRepository
from quant_trader.data.market_data import normalize_prices
from quant_trader.data.providers import YahooFinanceProvider, extract_corporate_actions
from quant_trader.data.storage import MarketDataStore
from quant_trader.domain.models import Instrument

logger = logging.getLogger(__name__)


def update_history(
    repository: InstrumentRepository,
    provider: YahooFinanceProvider,
    store: MarketDataStore,
    start: date,
    end: date,
    benchmark: Instrument | None = None,
    instrument_ids: list[str] | None = None,
) -> tuple[int, list[str]]:
    available = {
        instrument.instrument_id: instrument for instrument in repository.list_instruments()
    }
    if instrument_ids is None:
        instruments = list(available.values())
    else:
        unknown = set(instrument_ids).difference(available)
        if unknown:
            raise ValueError(f"Unknown configured instrument IDs: {sorted(unknown)}")
        instruments = [available[instrument_id] for instrument_id in instrument_ids]
    instruments += [benchmark] if benchmark is not None else []
    updated = 0
    failures: list[str] = []
    fx_cache: dict[str, object] = {}
    for instrument in instruments:
        try:
            raw = provider.fetch_history(instrument, start, end)
            retrieved_at = datetime.now(timezone.utc)
            store.save_raw(raw, instrument.instrument_id, provider.name, retrieved_at.isoformat())
            fx_rates = None
            if instrument.currency != "EUR":
                if instrument.currency not in fx_cache:
                    fx_cache[instrument.currency] = provider.fetch_fx_rates(
                        instrument.currency, start - timedelta(days=10), end
                    )
                fx_rates = fx_cache[instrument.currency]
            normalized = normalize_prices(
                raw,
                instrument,
                provider.name,
                retrieved_at=retrieved_at,
                fx_rates=fx_rates,
                price_adjustment=getattr(provider, "price_adjustment", "unadjusted"),
            )
            action_frame = extract_corporate_actions(raw, instrument)
            store.save(raw, normalized, action_frame, raw_already_saved=True)
            updated += len(normalized)
            logger.info("Stored %d daily bars for %s", len(normalized), instrument.instrument_id)
        except Exception as exc:
            failures.append(instrument.instrument_id)
            logger.exception("Unable to update %s: %s", instrument.instrument_id, exc)
    return updated, failures


def load_actions_as_domain(store: MarketDataStore, instrument_ids: list[str] | None = None):
    return actions_from_frame(store.load_actions(instrument_ids))
