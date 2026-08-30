import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def _records(df):
    clean = df.copy().replace({np.nan: None})
    return clean.to_dict(orient="records")


def _sanitize_open_offers(open_offers):
    allowed = [
        "player_id", "player_name", "market_value", "listed_price",
        "expires", "offer_count", "my_bid", "source",
    ]
    return [{key: offer.get(key) for key in allowed} for offer in (open_offers or [])]


def _write_history_snapshot(report):
    path = Path("reports/history.json")
    snapshots = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            snapshots = payload.get("snapshots", []) if isinstance(payload, dict) else []
        except Exception as exc:
            print(f"Warning: Could not read report history: {exc}")

    snapshot = {
        "generated_at": report["generated_at"],
        "manager_budgets": [
            {
                "user": row.get("User"),
                "budget": row.get("Budget"),
                "budget_confidence": row.get("Budget Confidence"),
                "team_value": row.get("Team Value"),
            }
            for row in report.get("manager_budgets", [])
        ],
        "market": [
            {
                "player_id": row.get("player_id"),
                "name": row.get("last_name"),
                "mv": row.get("mv"),
                "mv_change_yesterday": row.get("mv_change_yesterday"),
                "predicted_mv_target": row.get("predicted_mv_target"),
            }
            for row in report.get("market", [])
        ],
        "squad": [
            {
                "player_id": row.get("player_id"),
                "name": row.get("last_name"),
                "mv": row.get("mv"),
                "predicted_mv_target": row.get("predicted_mv_target"),
            }
            for row in report.get("squad", [])
        ],
        "my_open_offers": report.get("my_open_offers", []),
        "top_win_targets": report.get("strategy", {}).get("win_ranking", [])[:5],
    }

    snapshots.append(snapshot)
    snapshots = snapshots[-180:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"snapshots": snapshots}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_latest_report(
    league_name,
    metrics,
    manager_df,
    market_df,
    squad_df,
    transfer_df=None,
    bidding_df=None,
    open_offers=None,
    market_strategy_df=None,
    squad_signals_df=None,
    opponent_bid_profiles_df=None,
    points_profiles_df=None,
    win_ranking_df=None,
    ligainsider_signals_df=None,
):
    report = {
        "generated_at": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        "league": league_name,
        "model": metrics,
        "manager_budgets": _records(manager_df),
        "market": _records(market_df),
        "squad": _records(squad_df),
        "my_open_offers": _sanitize_open_offers(open_offers),
        "strategy": {
            "market_ranking": _records(market_strategy_df) if market_strategy_df is not None else [],
            "win_ranking": _records(win_ranking_df) if win_ranking_df is not None else [],
            "squad_signals": _records(squad_signals_df) if squad_signals_df is not None else [],
            "opponent_bid_profiles": _records(opponent_bid_profiles_df) if opponent_bid_profiles_df is not None else [],
        },
        "points_profiles": _records(points_profiles_df) if points_profiles_df is not None else [],
        "ligainsider_signals": _records(ligainsider_signals_df) if ligainsider_signals_df is not None else [],
        "transfer_history": _records(transfer_df) if transfer_df is not None else [],
        "manager_bidding_behavior": _records(bidding_df) if bidding_df is not None else [],
        "notes": {
            "opponent_budgets": "estimated; own budget marked exact",
            "open_offers": "only the authenticated user's visible outgoing bids are exposed on other managers' listings; competing bids are hidden by Kickbase",
            "bidding_behavior": "based on completed/winning transfers only; losing/open competing bids are not visible",
            "market_strategy": "capital-growth and bid-discipline layer based on MV model plus observed completed winning transfers",
            "expected_points": "transparent heuristic from Kickbase points/minutes, stabilized for small samples and adjusted by LigaInsider availability signals",
            "ligainsider": "public LigaInsider injury/suspension status and Topelf-page presence. Topelf presence can include an alternative and is therefore only a moderate positive signal, never a guaranteed start",
            "win_ranking": "combines expected next-match points, points-per-million, availability and capital-growth score; decision aid, not a guaranteed forecast",
            "s11_indicator": "raw upstream indicator retained without assuming an undocumented probability mapping",
        },
    }

    output = Path("reports/latest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_history_snapshot(report)
    return output
