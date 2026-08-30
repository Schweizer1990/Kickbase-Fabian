import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def _records(df):
    clean = df.copy().replace({np.nan: None})
    return clean.to_dict(orient="records")


def save_latest_report(
    league_name,
    metrics,
    manager_df,
    market_df,
    squad_df,
    transfer_df=None,
    bidding_df=None,
):
    """Write the latest analysis to a JSON file that ChatGPT can read from GitHub."""
    report = {
        "generated_at": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        "league": league_name,
        "model": metrics,
        "manager_budgets": _records(manager_df),
        "market": _records(market_df),
        "squad": _records(squad_df),
        "transfer_history": _records(transfer_df) if transfer_df is not None else [],
        "manager_bidding_behavior": _records(bidding_df) if bidding_df is not None else [],
        "notes": {
            "opponent_budgets": "estimated; own budget marked exact",
            "bidding_behavior": "based on completed/winning transfers only; losing/open bids are not visible",
        },
    }

    output = Path("reports/latest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output
