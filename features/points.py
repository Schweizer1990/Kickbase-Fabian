import math

import pandas as pd

from kickbase_api.config import BASE_URL, get_json_with_token


def _num(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _minutes(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("'", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _season_matches(season):
    matches = []
    for match in season.get("ph", []) if isinstance(season, dict) else []:
        points = _num(match.get("p"))
        minutes = _minutes(match.get("mp"))
        # Future matchdays usually have no points yet. A played matchday can
        # legitimately have 0 points and 0 minutes, so points-not-null is enough.
        if points is None and minutes <= 0:
            continue
        matches.append({"points": points or 0.0, "minutes": minutes})
    return matches


def _weighted_average(values):
    if not values:
        return 0.0
    weights = list(range(1, len(values) + 1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def _profile_from_performance(payload):
    seasons = payload.get("it", []) if isinstance(payload, dict) else []
    current = _season_matches(seasons[-1]) if seasons else []
    previous = _season_matches(seasons[-2]) if len(seasons) >= 2 else []

    current_recent = current[-5:]
    previous_recent = previous[-5:]

    current_minutes = [m["minutes"] for m in current_recent]
    previous_minutes = [m["minutes"] for m in previous_recent]

    current_points = [m["points"] for m in current_recent]
    previous_points = [m["points"] for m in previous_recent]

    def p90(matches):
        total_minutes = sum(m["minutes"] for m in matches)
        if total_minutes <= 0:
            return 0.0
        return sum(m["points"] for m in matches) / total_minutes * 90.0

    current_p90 = p90(current_recent)
    previous_p90 = p90(previous_recent)

    if len(current_recent) >= 3:
        blended_p90 = current_p90
        confidence = "high"
    elif len(current_recent) >= 1 and previous_recent:
        blended_p90 = current_p90 * 0.45 + previous_p90 * 0.55
        confidence = "medium"
    elif previous_recent:
        blended_p90 = previous_p90
        confidence = "low"
    else:
        blended_p90 = current_p90
        confidence = "low"

    if current_recent:
        expected_minutes = _weighted_average(current_minutes[-3:])
    elif previous_recent:
        expected_minutes = _weighted_average(previous_minutes[-3:])
    else:
        expected_minutes = 0.0

    expected_minutes = max(0.0, min(90.0, expected_minutes))
    expected_points = blended_p90 * expected_minutes / 90.0

    current_apps = len(current)
    current_total_points = sum(m["points"] for m in current)
    current_total_minutes = sum(m["minutes"] for m in current)

    return {
        "current_matchdays": current_apps,
        "current_points": round(current_total_points, 2),
        "current_minutes": round(current_total_minutes, 1),
        "recent_points_per_90": round(blended_p90, 2),
        "expected_minutes_next": round(expected_minutes, 1),
        "expected_points_next": round(expected_points, 2),
        "xp_confidence": confidence,
    }


def build_points_profiles(token, league_id, market_df, squad_df):
    """Build compact expected-points profiles for current squad and market players.

    Kickbase's league player performance endpoint provides matchday points and
    minutes. This is a transparent minutes/points heuristic, not a trained xP
    model yet. It becomes more reliable as the current season sample grows.
    """
    source = {}
    for label, df in (("market", market_df), ("squad", squad_df)):
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            player_id = row.get("player_id")
            if player_id is None:
                continue
            key = str(player_id)
            item = source.setdefault(key, {
                "player_id": key,
                "player_name": row.get("last_name"),
                "team": row.get("team_name"),
                "market_value": _num(row.get("mv")),
                "on_market": False,
                "in_squad": False,
            })
            item["on_market"] = item["on_market"] or label == "market"
            item["in_squad"] = item["in_squad"] or label == "squad"

    rows = []
    for player_id, base in source.items():
        try:
            url = f"{BASE_URL}/leagues/{league_id}/players/{player_id}/performance"
            payload = get_json_with_token(url, token)
            profile = _profile_from_performance(payload)
        except Exception as exc:
            print(f"Warning: Could not fetch league performance for player {player_id}: {exc}")
            profile = {
                "current_matchdays": 0,
                "current_points": 0.0,
                "current_minutes": 0.0,
                "recent_points_per_90": 0.0,
                "expected_minutes_next": 0.0,
                "expected_points_next": 0.0,
                "xp_confidence": "unavailable",
            }

        mv = base.get("market_value")
        xp = profile["expected_points_next"]
        ppm = xp / (mv / 1_000_000) if mv and mv > 0 else None
        rows.append({
            **base,
            **profile,
            "expected_points_per_million": round(ppm, 3) if ppm is not None else None,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["expected_points_next", "expected_points_per_million"],
        ascending=[False, False],
        ignore_index=True,
    )


def build_win_ranking(market_strategy_df, points_df):
    """Combine capital-growth and expected-points efficiency for transfer targets."""
    if market_strategy_df is None or market_strategy_df.empty:
        return pd.DataFrame()

    points_lookup = {}
    if points_df is not None and not points_df.empty:
        points_lookup = {
            str(row["player_id"]): row
            for _, row in points_df[points_df["on_market"] == True].iterrows()  # noqa: E712
        }

    rows = []
    for _, row in market_strategy_df.iterrows():
        player_id = str(row.get("player_id"))
        p = points_lookup.get(player_id, {})
        xp = _num(p.get("expected_points_next"), 0.0)
        xp_per_m = _num(p.get("expected_points_per_million"), 0.0)
        capital = _num(row.get("capital_score"), 0.0)
        # Capital remains important early in the season, but points efficiency
        # gets the larger weight because league points decide the championship.
        win_score = xp * 0.55 + xp_per_m * 0.25 + capital * 0.20
        rows.append({
            "player_id": player_id,
            "player_name": row.get("player_name"),
            "team": row.get("team"),
            "market_value": row.get("market_value"),
            "expected_points_next": round(xp, 2),
            "expected_points_per_million": round(xp_per_m, 3),
            "xp_confidence": p.get("xp_confidence", "unavailable"),
            "capital_score": row.get("capital_score"),
            "suggested_bid": row.get("suggested_bid"),
            "hard_max_bid": row.get("hard_max_bid"),
            "budget_fit": row.get("budget_fit"),
            "strategy_label": row.get("strategy_label"),
            "win_score": round(win_score, 3),
        })

    return pd.DataFrame(rows).sort_values(
        ["budget_fit", "win_score"], ascending=[False, False], ignore_index=True
    )
