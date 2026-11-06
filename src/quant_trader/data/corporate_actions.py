from __future__ import annotations

import pandas as pd

from quant_trader.domain.models import ActionType, CorporateAction


def actions_from_frame(frame: pd.DataFrame) -> list[CorporateAction]:
    actions: list[CorporateAction] = []
    for row in frame.itertuples(index=False):
        actions.append(
            CorporateAction(
                instrument_id=row.instrument_id,
                ex_date=pd.Timestamp(row.ex_date).date(),
                action_type=ActionType(row.action_type),
                value=float(row.value),
                currency=str(row.currency),
            )
        )
    return sorted(
        actions, key=lambda action: (action.ex_date, action.instrument_id, action.action_type)
    )
