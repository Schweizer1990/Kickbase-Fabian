"""Actual scoring, kept separate from xP and the current squad's point sums."""
import math
import re
from datetime import datetime, timezone

from kickbase_api.config import BASE_URL, get_json_with_token
from kickbase_api.manager import get_managers
from kickbase_api.user import get_username


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def current_season(payload):
    """Use season labels, never assume the API's season array order."""
    seasons = [s for s in payload.get("it", []) if isinstance(s, dict)]
    dated = []
    for season in seasons:
        label = str(season.get("sn") or season.get("ti") or "")
        years = re.findall(r"(?:19|20)\d{2}", label)
        if years:
            dated.append((int(years[0]), season))
    if dated:
        newest = max(year for year, _ in dated)
        candidates = [s for year, s in dated if year == newest]
        return candidates[0] if len(candidates) == 1 else None
    return seasons[0] if len(seasons) == 1 else None


def player_matchday_history(payload):
    season = current_season(payload)
    if season is None:
        return []
    rows = []
    for match in season.get("ph", []):
        if not isinstance(match, dict):
            continue
        rows.append({
            "season": season.get("ti") or season.get("sn"),
            "competition": season.get("n"),
            "matchday": number(match.get("day")),
            "date": match.get("md"),
            "points": number(match.get("p")),
            "minutes": number(str(match.get("mp", "")).replace("'", "")),
            "matchday_status": match.get("mdst"),
            "player_status": match.get("st"),
            "is_current": match.get("cur") is True,
        })
    return sorted(rows, key=lambda row: (row["matchday"] or -1, row["date"] or ""))


def completed_matchday(points_profiles_df, now=None):
    """Only anchor to explicit completed, dated player performances."""
    now = now or datetime.now(timezone.utc)
    days = []
    if points_profiles_df is None or points_profiles_df.empty:
        return None
    for history in points_profiles_df.get("matchday_history", []):
        for match in history:
            if number(match.get("matchday_status")) != 2 or match.get("matchday") is None:
                continue
            try:
                date = datetime.fromisoformat(match["date"].replace("Z", "+00:00"))
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                if date <= now:
                    days.append(match["matchday"])
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
    return max(days) if days else None


def manager_matchday_report(payload, manager_id, manager_name, completed_day=None):
    season = current_season(payload)
    base = {
        "source": "kickbase_manager_performance",
        "manager_id": str(manager_id), "manager_name": manager_name,
        "status": "available" if season is not None else "season_unresolved",
        "season": None, "total_points": None, "matchdays": [],
        "current_matchday": None, "latest_recorded_matchday": None,
        "latest_played_matchday": None,
        "historical_lineup_available": False,
    }
    if season is None:
        return base
    base.update(season=season.get("sn") or season.get("ti"), total_points=number(season.get("tp")))
    for entry in season.get("it", []):
        if not isinstance(entry, dict) or number(entry.get("day")) is None:
            continue
        base["matchdays"].append({
            "matchday": number(entry["day"]),
            "points": number(entry.get("mdp")),
            "is_current": entry.get("cur") is True,
            "matchday_win": entry.get("tw"),
        })
    base["matchdays"].sort(key=lambda row: row["matchday"])
    current = [row for row in base["matchdays"] if row["is_current"]]
    if len(current) == 1:
        base["current_matchday"] = current[0]
    # Keep future zero placeholders out of the latest result. Zero itself is a
    # valid result when an explicitly completed player matchday anchors it.
    if completed_day is not None:
        observed = [row for row in base["matchdays"]
                    if row["matchday"] <= completed_day and row["points"] is not None]
        if observed:
            base["latest_played_matchday"] = observed[-1]
    cutoff = current[0]["matchday"] if len(current) == 1 else completed_day
    recorded = [row for row in base["matchdays"] if row["points"] is not None
                and row["points"] != 0 and cutoff is not None and row["matchday"] <= cutoff]
    candidates = recorded[-1:]
    if base["latest_played_matchday"] is not None:
        candidates.append(base["latest_played_matchday"])
    base["latest_recorded_matchday"] = max(candidates, key=lambda row: row["matchday"]) if candidates else None
    return base


def build_own_matchday_report(token, league_id, points_profiles_df):
    try:
        own_name = get_username(token)
        matches = [(name, manager_id) for name, manager_id in get_managers(token, league_id) if name == own_name]
        if len(matches) != 1:
            return {"status": "manager_unresolved", "matchdays": []}
        name, manager_id = matches[0]
        payload = get_json_with_token(f"{BASE_URL}/leagues/{league_id}/managers/{manager_id}/performance", token)
        return manager_matchday_report(payload, manager_id, name, completed_matchday(points_profiles_df))
    except Exception:
        # Optional scoring must not break transfers, forecasts or expose tokens.
        print("Warning: Actual manager matchday points unavailable.")
        return {"status": "fetch_failed", "matchdays": [], "total_points": None}
