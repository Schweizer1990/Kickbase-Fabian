"""Conservative read-only matchday and trading decision support.

No automatic bids or sales. Do not interpret absence from an injury list as
medical clearance. The points model already includes starter probability.
"""
import itertools
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Zurich")
FORMATIONS = ((3,4,3), (4,4,2), (4,3,3), (3,5,2),
              (5,3,2), (4,5,1), (5,4,1), (5,2,3))
OUT_STATUSES = {"injured", "out", "suspended"}
RISK_FACTORS = {"injured": 0.0, "out": 0.0, "suspended": 0.0,
                "likely_out": 0.15, "rehab": 0.45,
                "doubtful": 0.60, "slight_knock": 0.82}


def numeric(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def deadline(now):
    days = (4 - now.weekday()) % 7
    target = (now + timedelta(days=days)).replace(hour=20, minute=30,
                                                  second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=7)
    return target


def load_overrides(path="reports/availability_overrides.json"):
    """Optional manually verified statuses; never auto-infer from rumours."""
    file = Path(path)
    if not file.exists():
        return {}
    data = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("availability_overrides must be an object keyed by player ID")
    return data


def build_winning_edge(report, overrides=None):
    now = datetime.fromisoformat(report["generated_at"]).astimezone(TZ)
    target = deadline(now)
    squad = report.get("squad") or []
    points = {str(p.get("player_id")): p for p in report.get("points_profiles", [])}
    statuses = {str(p.get("player_id")): p for p in report.get("ligainsider_signals", [])}
    overrides = overrides if overrides is not None else load_overrides()
    rows = []
    for player in squad:
        pid = str(player.get("player_id"))
        profile, status = points.get(pid, {}), statuses.get(pid, {})
        verified = overrides.get(pid)
        state = str(status.get("ligainsider_status") or "unknown")
        # "available" is merely absence from the scraped injury table.
        source = "ligainsider_structured_injury_table" if state not in ("available", "unknown") else "unconfirmed"
        factor = numeric(status.get("availability_factor"), 1.0) if state in RISK_FACTORS else 1.0
        if isinstance(verified, dict):
            until = verified.get("expires_at")
            active = not until or datetime.fromisoformat(until).astimezone(TZ) > now
            if active and verified.get("status") in (set(RISK_FACTORS) | {"available"}):
                state = verified["status"]
                factor = RISK_FACTORS.get(state, 1.0)
                source = "manual_override"
        xp = max(0, numeric(profile.get("expected_points_next")))
        adjusted = round(xp * factor, 2)
        delta = numeric(player.get("predicted_mv_target"))
        previous = numeric(player.get("mv_change_yesterday"))
        momentum = ("declining" if delta < 0 and previous <= 0 else
                    "reversal_watch" if delta < 0 else
                    "growing" if delta > 0 and previous > 0 else "mixed")
        if state in OUT_STATUSES:
            flag = "confirmed_unavailable"
        elif state in RISK_FACTORS or source == "manual_override" and state != "available":
            flag = "availability_risk"
        else:
            flag = "not_medically_verified"
        # Strong momentum is valuable even if the player is unlikely to start.
        # Only prefer a sell after both trend and sporting value are poor.
        if delta < 0 and xp < 25 and previous <= 0:
            action = "sell_candidate"
        elif delta < 0 and xp < 25:
            action = "sell_watch"
        elif delta < 0:
            action = "protect_points_watch_trend"
        elif delta > 0 and xp < 25:
            action = "hold_for_trading"
        else:
            action = "hold"
        rows.append({
            "player_id": pid, "player_name": player.get("last_name"),
            "position": player.get("position"), "market_value": player.get("mv"),
            "expected_points_model": round(xp, 2), "expected_points_risk_adjusted": adjusted,
            "availability": state, "availability_evidence": source,
            "availability_flag": flag, "mv_change_last": previous,
            "mv_change_predicted": delta, "momentum": momentum,
            "decision": action,
        })
    # Enumerate only legal formation shapes and keep exactly one GK.
    candidates = {pos: sorted(
        [p for p in rows if p["position"] == pos],
        key=lambda p: p["expected_points_risk_adjusted"], reverse=True)
        for pos in (1, 2, 3, 4)}
    best = None
    for defence, midfield, attack in FORMATIONS:
        counts = {1: 1, 2: defence, 3: midfield, 4: attack}
        if any(len(candidates[pos]) < count for pos, count in counts.items()):
            continue
        picked = [p for pos, count in counts.items() for p in candidates[pos][:count]]
        total = round(sum(p["expected_points_risk_adjusted"] for p in picked), 2)
        if best is None or total > best["expected_points"]:
            best = {"formation": f"{defence}-{midfield}-{attack}",
                    "expected_points": total,
                    "players": [{"player_id": p["player_id"], "name": p["player_name"],
                                 "position": p["position"],
                                 "expected_points": p["expected_points_risk_adjusted"],
                                 "availability_flag": p["availability_flag"]}
                                for p in picked]}
    unverified = [p["player_name"] for p in (best or {}).get("players", [])
                  if p["availability_flag"] != "confirmed_unavailable"
                  and p["availability_evidence"] == "unconfirmed"]
    budget_rows = [r for r in report.get("manager_budgets", [])
                   if r.get("Budget Confidence") == "exact"]
    cash = numeric(budget_rows[0].get("Budget")) if budget_rows else None
    open_offers = report.get("my_open_offers", [])
    return {
        "generated_at": report["generated_at"],
        "deadline": target.isoformat(),
        "deadline_hours_remaining": round((target-now).total_seconds()/3600, 2),
        "cash_now": cash, "positive_cash_now": cash is not None and cash >= 0,
        "cash_to_recover_for_deadline": max(0, -cash) if cash is not None else None,
        "open_bid_commitment": sum(numeric(b.get("my_bid")) for b in open_offers),
        "squad_size": len(squad), "free_slots": max(0, 15-len(squad)),
        "lineup": best, "lineup_unverified_players": unverified,
        "player_decisions": rows,
        "sell_watch": [r["player_name"] for r in rows if r["decision"] in ("sell_candidate", "sell_watch")],
        "trading_holds": [r["player_name"] for r in rows if r["decision"] == "hold_for_trading"],
        "notes": [
            "No player is medically cleared solely because an injury page omitted them.",
            "Expected points are heuristic and may be stale when breaking injury news emerges.",
            "No automatic transfer or lineup changes; Friday 20:30 Europe/Zurich cash deadline.",
            "Existing auction scenarios and club limits remain in strategy.decision_plan.",
        ],
    }
