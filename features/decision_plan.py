"""Read-only auction pricing, replacement comparisons and squad-slot scenarios.

Multi-update growth is a discounted scenario, not a trained multi-day forecast.
No function in this module sends bids, sells players or changes a lineup.
"""
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULTS = {
    "max_squad_size": 15, "max_players_per_club": 2,
    "market_update_hour": 22, "market_update_minute": 0,
    "holding_updates": 3, "growth_discount": 0.75,
    "max_replacements": 2, "max_expected_points_loss": 15,
    "points_core_threshold": 60,
    "protected_player_ids": ["49", "1920", "1873", "3284"],
}
TZ = ZoneInfo("Europe/Zurich")


def number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(TZ) if result.tzinfo else None
    except (ValueError, TypeError):
        return None


def load_config(path="strategy_config.json"):
    config = dict(DEFAULTS)
    if Path(path).exists():
        config.update(json.loads(Path(path).read_text(encoding="utf-8")))
    if not 0 < config["growth_discount"] <= 1:
        raise ValueError("growth_discount must be in (0, 1]")
    if config["holding_updates"] < 1 or config["max_squad_size"] < 1:
        raise ValueError("holding_updates and max_squad_size must be positive")
    if config["max_players_per_club"] < 1 or config["max_replacements"] < 0:
        raise ValueError("Invalid squad limits")
    return config


def update_count(now, expiry, config):
    """Count scheduled updates strictly between the snapshot and auction end."""
    if expiry is None or expiry <= now:
        return None
    next_update = now.replace(hour=config["market_update_hour"],
                              minute=config["market_update_minute"], second=0, microsecond=0)
    if next_update <= now:
        next_update += timedelta(days=1)
    count = 0
    while next_update < expiry:
        count += 1
        next_update += timedelta(days=1)
    return count


def growth_scenario(delta, updates, discount):
    return sum(delta * discount ** day for day in range(updates))


def simulate_auctions(squad, bids, sales=(), config=None):
    """Assume every listed bid wins; expose cancellations at 15, not win odds."""
    config = config or DEFAULTS
    sales = {str(x) for x in sales}
    owned = {str(x["player_id"]): x for x in squad if str(x["player_id"]) not in sales}
    clubs = Counter(x.get("team_name") for x in owned.values())
    events, cancelled, unresolved = [], [], []
    dated = []
    for bid in bids:
        expiry = timestamp(bid.get("expires_at"))
        if expiry is None:
            unresolved.append(str(bid["player_id"]))
        else:
            dated.append((expiry, bid))
    dated.sort(key=lambda item: (item[0], str(item[1]["player_id"])))
    ambiguous = len({date for date, _ in dated}) != len(dated)
    for _, bid in dated:
        pid, team = str(bid["player_id"]), bid.get("team")
        if pid in owned:
            continue
        if len(owned) >= config["max_squad_size"]:
            cancelled.append(pid)
            events.append({"player_id": pid, "status": "cancelled_squad_full"})
            continue
        if not team:
            unresolved.append(pid)
            continue
        if clubs[team] >= config["max_players_per_club"]:
            events.append({"player_id": pid, "status": "blocked_club_limit"})
            continue
        owned[pid] = bid
        clubs[team] += 1
        events.append({"player_id": pid, "status": "assumed_win", "squad_size": len(owned)})
    return {"assumption": "all_bids_win_sales_complete_before_first_auction",
            "events": events, "cancelled_bid_ids": cancelled,
            "unresolved_bid_ids": unresolved, "ambiguous_expiry_order": ambiguous,
            "final_squad_size": len(owned),
            "safe": not cancelled and not unresolved and not ambiguous
                    and all(e["status"] == "assumed_win" for e in events)}


def build_decision_plan(report, config=None):
    config = config or load_config()
    now = timestamp(report.get("market_fetched_at"))
    if now is None:
        return {"status": "unavailable", "reason": "missing_market_snapshot_timestamp"}
    squad = report.get("squad", [])
    market = {str(x["player_id"]): x for x in report.get("market", [])}
    points = {str(x["player_id"]): x for x in report.get("points_profiles", [])}
    guardrails = {str(x["player_id"]): x for x in report.get("strategy", {}).get("bid_guardrails", [])}
    ranking = {str(x["player_id"]): x for x in report.get("strategy", {}).get("market_ranking", [])}
    owned_ids = {str(x["player_id"]) for x in squad}
    club_counts = Counter(x.get("team_name") for x in squad)
    budget_rows = [x for x in report.get("manager_budgets", []) if x.get("Budget Confidence") == "exact"]
    budget = number(budget_rows[0].get("Budget")) if budget_rows else None
    bids = []
    pricing = []
    for live in report.get("market_live", []):
        pid = str(live["player_id"])
        player = market.get(pid, {})
        team = player.get("team_name")
        expiry = timestamp(live.get("expires_at"))
        count = update_count(now, expiry, config)
        mv = number(live.get("market_value"))
        historical_mv = number(player.get("mv"))
        delta = number(player.get("predicted_mv_target"))
        own_bid = number(live.get("my_bid"))
        if own_bid is not None and own_bid > 0:
            bids.append({"player_id": pid, "team": team, "my_bid": own_bid,
                         "expires_at": live.get("expires_at"), "source": "existing"})
        reason = None
        if live.get("is_own_listing") or pid in owned_ids:
            reason = "own_player"
        elif live.get("seller_id") or live.get("seller_name"):
            reason = "manager_listing_excluded"
        elif count is None:
            reason = "unknown_or_expired_auction"
        elif mv is None or mv <= 0 or delta is None or historical_mv is None:
            reason = "missing_prediction_or_market_value"
        elif abs(mv - historical_mv) > 1:
            reason = "prediction_market_value_mismatch_refresh_required"
        elif delta <= 0 or number(player.get("mv_change_yesterday"), -1) < 0:
            reason = "non_positive_trend"
        elif club_counts[team] >= config["max_players_per_club"]:
            reason = "club_limit_requires_sale"
        elif not team:
            reason = "unknown_club"
        row = {"player_id": pid, "player_name": live.get("player_name"), "team": team,
               "market_value": mv, "expires_at": live.get("expires_at"),
               "updates_before_expiry": count, "eligible": reason is None, "reason": reason,
               "existing_bid": own_bid, "suggested_bid": None, "hard_max_bid": None}
        if mv is not None and delta is not None and count is not None:
            projected = mv + growth_scenario(delta, count, config["growth_discount"])
            premium = number(guardrails.get(pid, {}).get("league_p75_overpay_pct"))
            if premium is None:
                premium = number(ranking.get(pid, {}).get("league_median_overpay_pct"), 0)
            premium = max(0, min(premium, 25))
            post_growth = growth_scenario(delta * config["growth_discount"] ** count,
                                          config["holding_updates"], config["growth_discount"])
            maximum = math.floor(projected + max(post_growth, 0) * 0.5)
            suggested = min(math.ceil(projected * (1 + premium / 100)), maximum)
            row.update({"projected_mv_at_expiry": round(projected), "competition_premium_pct": premium,
                        "post_purchase_growth_scenario": round(post_growth),
                        "pricing_scenario": "daily_prediction_discounted_each_future_update",
                        "suggested_bid": suggested if reason is None else None,
                        "hard_max_bid": maximum if reason is None else None,
                        "scenario_profit_after_holding": round(projected + post_growth - (own_bid or suggested))})
        pricing.append(row)

    # Only profitable traders with known points and a completed purchase can be replaced.
    costs = {}
    for transfer in report.get("transfer_history", []):
        if budget_rows and transfer.get("buyer") == budget_rows[0].get("User"):
            price = number(transfer.get("price"))
            if price is not None:
                costs[str(transfer["player_id"])] = price
    protected = {str(x) for x in config["protected_player_ids"]}
    comparisons = []
    for sell in squad:
        sid = str(sell["player_id"])
        sell_xp = number(points.get(sid, {}).get("expected_points_next"))
        sell_mv, sell_delta = number(sell.get("mv")), number(sell.get("predicted_mv_target"))
        if (sid in protected or sell_xp is None or sell_xp >= config["points_core_threshold"]
                or sell_mv is None or sell_delta is None or sid not in costs or sell_mv <= costs[sid]):
            continue
        retained_growth = growth_scenario(sell_delta, config["holding_updates"], config["growth_discount"])
        for buy in pricing:
            xp = number(points.get(buy["player_id"], {}).get("expected_points_next"))
            profile = points.get(buy["player_id"], {})
            if (not buy["eligible"] or buy["existing_bid"] or xp is None
                    or profile.get("xp_confidence") == "unavailable"
                    or profile.get("role_signal") in ("unknown", "out")
                    or xp < sell_xp - config["max_expected_points_loss"]):
                continue
            # Longer waits also incur growth forgone on the sold player before acquisition.
            retained = growth_scenario(sell_delta, config["holding_updates"] + buy["updates_before_expiry"], config["growth_discount"])
            advantage = buy["scenario_profit_after_holding"] - retained
            if advantage > 0:
                comparisons.append({"sell_player_id": sid, "sell_player_name": sell.get("last_name"),
                                    "sell_market_value": sell_mv, "profit_vs_purchase": round(sell_mv - costs[sid]),
                                    "buy_player_id": buy["player_id"], "buy_player_name": buy["player_name"],
                                    "suggested_bid": buy["suggested_bid"], "expires_at": buy["expires_at"],
                                    "expected_points_change": round(xp - sell_xp, 2),
                                    "incremental_growth_scenario": round(advantage),
                                    "retained_growth_scenario": round(retained_growth)})
    comparisons.sort(key=lambda row: row["incremental_growth_scenario"], reverse=True)
    replacements, sales, targets = [], set(), set()
    for row in comparisons:
        if len(replacements) >= config["max_replacements"]:
            break
        if row["sell_player_id"] in sales or row["buy_player_id"] in targets:
            continue
        candidate_sales = sales | {row["sell_player_id"]}
        target = next(x for x in pricing if x["player_id"] == row["buy_player_id"])
        candidate_bids = bids + [{"player_id": x["buy_player_id"],
                                  "team": market[x["buy_player_id"]].get("team_name"),
                                  "my_bid": x["suggested_bid"], "expires_at": x["expires_at"],
                                  "source": "proposed"} for x in replacements + [row]]
        available = None if budget is None else budget + sum(number(x.get("mv"), 0) for x in squad if str(x["player_id"]) in candidate_sales)
        if available is None or sum(x["my_bid"] for x in candidate_bids) > available:
            continue
        if not simulate_auctions(squad, candidate_bids, candidate_sales, config)["safe"]:
            continue
        replacements.append(row)
        sales, targets = candidate_sales, targets | {target["player_id"]}
    proposed = [{"player_id": row["buy_player_id"], "team": market[row["buy_player_id"]].get("team_name"),
                 "my_bid": row["suggested_bid"], "expires_at": row["expires_at"], "source": "proposed"}
                for row in replacements]
    return {"status": "ready", "config": config, "auction_bids": pricing,
            "replacement_comparisons": comparisons, "recommended_replacements": replacements,
            "existing_bids_scenario": simulate_auctions(squad, bids, config=config),
            "proposed_bids_without_sales_scenario": simulate_auctions(squad, bids + proposed, config=config),
            "recommended_sales_player_ids": sorted(sales),
            "recommended_plan_scenario": simulate_auctions(squad, bids + proposed, sales, config),
            "notes": ["No automatic bids or sales. Prices and gains are scenarios, not guaranteed profits.",
                      "Update time is configurable; unknown expiry, stale MV, club limits and manager listings block new recommendations.",
                      "Expected points are a heuristic. This comparison does not certify a legal or improved starting XI."]}
