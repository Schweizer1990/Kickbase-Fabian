import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from kickbase_api.league import get_league_activities
from kickbase_api.player import get_player_market_value


def _transfer_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return str(value)[:10]


def _market_value_lookup(token, competition_id, player_ids):
    """Fetch daily market-value histories once per traded player."""
    result = {}
    for player_id in sorted({str(pid) for pid in player_ids if pid is not None}):
        try:
            history = get_player_market_value(token, competition_id, player_id, 365)
            result[player_id] = {item["date"]: item["mv"] for item in history}
        except Exception as exc:
            print(f"Warning: Could not fetch market-value history for player {player_id}: {exc}")
            result[player_id] = {}
    return result


def build_transfer_history(token, league_id, league_start_date, competition_id=1):
    """Build a completed-transfer ledger with market value and overpay metrics.

    The activity feed contains completed transfers, not losing/open bids. Existing
    rows are merged from reports/transfers.json so the local history survives even
    when older entries eventually disappear from the bounded activity feed.
    """
    activities, _, _ = get_league_activities(token, league_id, league_start_date)
    mv_history = _market_value_lookup(token, competition_id, [a.get("pi") for a in activities])

    rows = []
    for transfer in activities:
        player_id = transfer.get("pi")
        date = _transfer_date(transfer.get("dt"))
        price = transfer.get("trp")
        buyer = transfer.get("byr")
        seller = transfer.get("slr")
        transfer_id = transfer.get("tid")

        try:
            price = float(price) if price is not None else None
        except (TypeError, ValueError):
            price = None

        market_value = mv_history.get(str(player_id), {}).get(date) if date else None
        try:
            market_value = float(market_value) if market_value is not None else None
        except (TypeError, ValueError):
            market_value = None

        overpay = None
        overpay_pct = None
        if price is not None and market_value not in (None, 0):
            overpay = price - market_value
            overpay_pct = (overpay / market_value) * 100

        if buyer and seller:
            transfer_type = "manager_to_manager"
        elif buyer:
            transfer_type = "kickbase_purchase"
        elif seller:
            transfer_type = "kickbase_sale"
        else:
            transfer_type = "unknown"

        rows.append({
            "transfer_id": transfer_id,
            "timestamp": transfer.get("dt"),
            "date": date,
            "player_id": str(player_id) if player_id is not None else None,
            "player_name": transfer.get("pn"),
            "buyer": buyer,
            "seller": seller,
            "price": price,
            "market_value_at_transfer": market_value,
            "overpay": overpay,
            "overpay_pct": round(overpay_pct, 2) if overpay_pct is not None else None,
            "type": transfer_type,
        })

    history_path = Path("reports/transfers.json")
    existing = []
    if history_path.exists():
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            existing = payload.get("transfers", payload if isinstance(payload, list) else [])
        except Exception as exc:
            print(f"Warning: Could not read existing transfer history: {exc}")

    combined = existing + rows
    deduped = {}
    for row in combined:
        # Do not assume Kickbase's `tid` is globally unique. The composite key is
        # stable for one completed transfer and prevents accidental row collapse.
        key = "|".join(str(row.get(k)) for k in ["timestamp", "player_id", "buyer", "seller", "price"])
        deduped[key] = row

    history = list(deduped.values())
    history.sort(key=lambda item: item.get("timestamp") or "")

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps({"updated_at": datetime.now().isoformat(), "transfers": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return pd.DataFrame(history)


def summarize_manager_bidding(transfer_df):
    """Summarize observed completed purchases per manager.

    This measures winning/completed transfer prices only. It does not reveal
    losing bids, which Kickbase does not expose through this data source.
    """
    columns = [
        "Manager", "Purchases", "Priced Purchases", "Total Spend",
        "Average Overpay %", "Median Overpay %", "Max Overpay %"
    ]
    if transfer_df.empty:
        return pd.DataFrame(columns=columns)

    purchases = transfer_df[transfer_df["buyer"].notna()].copy()
    if purchases.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for manager, group in purchases.groupby("buyer"):
        priced = group[group["overpay_pct"].notna()].copy()
        rows.append({
            "Manager": manager,
            "Purchases": int(len(group)),
            "Priced Purchases": int(len(priced)),
            "Total Spend": float(group["price"].fillna(0).sum()),
            "Average Overpay %": round(float(priced["overpay_pct"].mean()), 2) if not priced.empty else None,
            "Median Overpay %": round(float(priced["overpay_pct"].median()), 2) if not priced.empty else None,
            "Max Overpay %": round(float(priced["overpay_pct"].max()), 2) if not priced.empty else None,
        })

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(["Purchases", "Total Spend"], ascending=False, ignore_index=True)
